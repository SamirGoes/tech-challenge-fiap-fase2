output "function_url" {
  description = "URL pública HTTPS da API (endpoints /health e /predict)"
  value       = aws_apigatewayv2_stage.default.invoke_url
}

output "ecr_repository_url" {
  description = "URL do repositório ECR — usar para fazer push da imagem Docker antes do primeiro deploy"
  value       = aws_ecr_repository.api.repository_url
}
