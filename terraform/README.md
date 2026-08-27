# Deploy da API na AWS (Lambda + API Gateway)

Pré-requisitos: conta AWS configurada localmente (`aws configure`), Terraform >= 1.5, Docker rodando.

## Arquitetura

- **ECR**: guarda a imagem Docker da API (equivalente ao Artifact Registry do GCP).
- **Lambda (imagem de container)**: roda a API FastAPI via Mangum (equivalente ao serviço do Cloud Run).
- **API Gateway (HTTP API)**: expõe a Lambda publicamente via HTTPS, com uma rota `$default` (proxy total) apontando pra Lambda.
- **SSM Parameter Store (SecureString)**: guarda a chave da API Anthropic — usado em vez de Secrets Manager porque é gratuito (Secrets Manager cobra ~US$0,40/segredo/mês).
- **CloudWatch Logs**: logs da Lambda.

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
`apigateway:*` (ex.: policy gerenciada `AmazonAPIGatewayAdministrator`), além das permissões de
Lambda, ECR, IAM e SSM já necessárias antes.

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

## 3. Provisionar o resto (Lambda, API Gateway, SSM, logs)

```bash
cd terraform
terraform apply -var-file=terraform.tfvars
terraform output function_url
```

## 4. Testar

```bash
FUNCTION_URL=$(terraform output -raw function_url)
curl "${FUNCTION_URL}health"
curl "${FUNCTION_URL}status"
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
