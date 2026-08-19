# Deploy da API na AWS (Lambda + Function URL)

Pré-requisitos: conta AWS configurada localmente (`aws configure`), Terraform >= 1.5, Docker rodando.

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

docker build -t "$REPO_URL:latest" .
docker push "$REPO_URL:latest"
```

## 3. Provisionar o resto (Lambda, Function URL, SSM, logs)

```bash
cd terraform
terraform apply -var-file=terraform.tfvars
terraform output function_url
```

## 4. Testar

```bash
FUNCTION_URL=$(terraform output -raw function_url)
curl "${FUNCTION_URL}health"
```

## Destruir tudo (evitar custo residual)

```bash
terraform destroy -var-file=terraform.tfvars
```

## Segredo da API Anthropic

Nunca commitar a chave real. Use `terraform.tfvars` (já no `.gitignore`) ou a variável de ambiente `TF_VAR_anthropic_api_key` — o valor só é escrito no SSM Parameter Store (`SecureString`, criptografado), nunca no código ou na imagem Docker.
