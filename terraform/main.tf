terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# --- Repositório de imagem Docker (equivalente ao Artifact Registry do GCP) ---
resource "aws_ecr_repository" "api" {
  name                 = "${var.nome_projeto}-api"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
}

# --- Segredo da API Anthropic ---
# SSM Parameter Store (SecureString) em vez de Secrets Manager: mesma função,
# mas gratuito — Secrets Manager cobra ~US$0,40/segredo/mês.
resource "aws_ssm_parameter" "anthropic_api_key" {
  name  = "/fase2/anthropic-api-key"
  type  = "SecureString"
  value = var.anthropic_api_key
}

# --- Permissões da Lambda ---
data "aws_iam_policy_document" "lambda_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda_exec" {
  name               = "${var.nome_projeto}-lambda-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json
}

resource "aws_iam_role_policy_attachment" "lambda_basic_logs" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "lambda_ssm_read" {
  statement {
    actions   = ["ssm:GetParameter"]
    resources = [aws_ssm_parameter.anthropic_api_key.arn]
  }
}

resource "aws_iam_role_policy" "lambda_ssm_read" {
  name   = "${var.nome_projeto}-ssm-read"
  role   = aws_iam_role.lambda_exec.id
  policy = data.aws_iam_policy_document.lambda_ssm_read.json
}

# --- Função Lambda (imagem de container — equivalente ao serviço do Cloud Run) ---
resource "aws_lambda_function" "api" {
  function_name = "${var.nome_projeto}-api"
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.api.repository_url}:latest"

  timeout     = 30
  memory_size = 1024

  environment {
    variables = {
      ANTHROPIC_API_KEY_PARAM = aws_ssm_parameter.anthropic_api_key.name
    }
  }

  depends_on = [aws_iam_role_policy_attachment.lambda_basic_logs]
}

# --- Endpoint HTTPS público via API Gateway (HTTP API) ---
# Trocado de Lambda Function URL para API Gateway porque contas AWS novas
# têm uma restrição de conta que bloqueia invocação anônima (403) em Function
# URLs com authorization_type = NONE, mesmo com a resource policy correta.
# O API Gateway não tem essa mesma restrição.
resource "aws_apigatewayv2_api" "api" {
  name          = "${var.nome_projeto}-api"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "lambda" {
  api_id                 = aws_apigatewayv2_api.api.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.api.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "default" {
  api_id    = aws_apigatewayv2_api.api.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.api.id
  name        = "$default"
  auto_deploy = true
}

resource "aws_lambda_permission" "apigw_invoke" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.api.execution_arn}/*/*"
}

# --- Logs (satisfaz o requisito de monitoramento/logging do enunciado) ---
resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/${aws_lambda_function.api.function_name}"
  retention_in_days = 14
}
