# Deploy da API + Frontend na AWS

Pré-requisitos: conta AWS configurada localmente (`aws configure`), Terraform >= 1.5, Docker rodando, Node.js/npm (pra buildar o frontend Angular).

## Arquitetura

- **ECR**: guarda a imagem Docker da API (equivalente ao Artifact Registry do GCP).
- **Lambda (imagem de container)**: roda a API FastAPI via Mangum (equivalente ao serviço do Cloud Run).
- **API Gateway (HTTP API)**: expõe a Lambda publicamente via HTTPS, com uma rota `$default` (proxy total) apontando pra Lambda.
- **SSM Parameter Store (SecureString)**: guarda a chave da API Anthropic — usado em vez de Secrets Manager porque é gratuito (Secrets Manager cobra ~US$0,40/segredo/mês).
- **CloudWatch Logs**: logs da Lambda.
- **S3 (bucket privado)**: guarda o build estático do frontend Angular.
- **CloudFront**: serve o frontend a partir do S3 e encaminha as rotas `/predict*`, `/health` e `/status` pro API Gateway, tudo no mesmo domínio.

### Por que CloudFront na frente de tudo (front + API no mesmo domínio)

O frontend Angular chama a API com paths relativos (`/predict`, `/health`, etc — ver
`frontend/src/app/api.ts`), sem domínio fixo. Em vez de hospedar o front num domínio e a API
noutro (o que exigiria configurar CORS em produção e um `environment.ts` com a URL da API), o
CloudFront resolve isso com **cache behaviors por path**: a origem padrão serve os arquivos
estáticos do S3, e comportamentos específicos pra `/predict*`, `/health` e `/status` encaminham
pro API Gateway. Do ponto de vista do navegador, tudo é same-origin — zero mudança de código no
front e zero CORS pra configurar.

### Por que API Gateway em vez de Lambda Function URL

A primeira versão desse Terraform expunha a Lambda direto via `aws_lambda_function_url` com
`authorization_type = "NONE"`. Isso funciona normalmente, mas **contas AWS novas/Free Tier têm uma
restrição de conta que bloqueia invocação anônima em Function URLs** (retorna `403 Forbidden` mesmo
com a resource policy certa liberando `principal = "*"`). Essa restrição é um guard-rail
anti-abuso da AWS e não é visível/ajustável via Terraform ou CLI — só é resolvida abrindo um caso
gratuito no AWS Support (Account & Billing) pedindo a remoção.

Como alternativa mais rápida e sem custo, trocamos o Function URL por um **API Gateway HTTP API**
na frente da mesma Lambda (`aws_apigatewayv2_api` + `aws_apigatewayv2_integration` +
`aws_apigatewayv2_route` + `aws_apigatewayv2_stage`). O API Gateway não tem essa mesma restrição de
conta nova. O formato do evento que a Lambda recebe (payload format version 2.0) é o mesmo que o
Function URL usaria, então o código da API (`api/main.py`, via Mangum) não precisou de nenhuma
alteração.

**Permissão IAM necessária**: o usuário/role que roda o `terraform apply` precisa de permissão
`apigateway:*` (ex.: policy gerenciada `AmazonAPIGatewayAdministrator`), `cloudfront:*`
(`CloudFrontFullAccess`) e `s3:*` (`AmazonS3FullAccess`), além das permissões de Lambda, ECR, IAM e
SSM já necessárias antes.

## 1. Criar o repositório ECR e ler a URL dele

O ECR precisa existir *antes* do primeiro push da imagem, mas a Lambda (passo 3) precisa de uma imagem já publicada no ECR para ser criada — por isso o primeiro `apply` roda só até a Lambda falhar (esperado), ou usa-se `-target` para criar só o ECR primeiro:

```bash
cd terraform
terraform init
terraform apply -target=aws_ecr_repository.api -var-file=terraform.tfvars
terraform output ecr_repository_url
```

## 2. Build e push da imagem Docker

```bash
cd ..
REPO_URL=$(cd terraform && terraform output -raw ecr_repository_url)
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$REPO_URL"

docker build -t "${REPO_URL}:latest" .
docker push "${REPO_URL}:latest"
```

> **Nota (zsh):** use sempre `${REPO_URL}:latest` com chaves, não `$REPO_URL:latest`. Em zsh,
> `$VAR:l` é interpretado como um modificador de histórico (`:l` = lowercase), o que corrompe a tag
> silenciosamente (ex.: `...ga-api:latest` vira `...ga-apiatest`). Com chaves esse problema não
> ocorre.

## 3. Provisionar o resto (Lambda, API Gateway, S3, CloudFront, SSM, logs)

```bash
cd terraform
terraform apply -var-file=terraform.tfvars
```

A criação da distribuição CloudFront demora de 5 a 15 minutos pra propagar globalmente — o
`apply` só retorna depois disso.

## 4. Build e upload do frontend

```bash
cd ../frontend
npm install
npm run build
cd ..

BUCKET=$(cd terraform && terraform output -raw frontend_bucket)
DIST_ID=$(cd terraform && terraform output -raw cloudfront_distribution_id)

aws s3 sync frontend/dist/frontend/browser/ "s3://${BUCKET}/" --delete
aws cloudfront create-invalidation --distribution-id "$DIST_ID" --paths "/*"
```

Repita esse passo (sync + invalidation) a cada novo build do front — o Terraform não faz isso
automaticamente.

## 5. Testar

```bash
cd terraform
FRONTEND_URL=$(terraform output -raw frontend_url)
curl "${FRONTEND_URL}/health"
curl "${FRONTEND_URL}/status"
open "$FRONTEND_URL"   # abre o front no navegador (macOS)
```

## Destruir tudo (evitar custo residual)

```bash
terraform destroy -var-file=terraform.tfvars
```

## Segredo da API Anthropic

Nunca commitar a chave real. Use `terraform.tfvars` (já no `.gitignore`) ou a variável de ambiente `TF_VAR_anthropic_api_key` — o valor só é escrito no SSM Parameter Store (`SecureString`, criptografado), nunca no código ou na imagem Docker.

Se `anthropic_api_key` não for uma chave válida (ex.: placeholder), a API continua funcionando
normalmente para `/predict` — só o campo `explicacao` da resposta vem com uma mensagem de erro em
vez do texto gerado pela LLM.
