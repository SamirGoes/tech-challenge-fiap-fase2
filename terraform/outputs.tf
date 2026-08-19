output "function_url" {
  description = "URL pública HTTPS da API (endpoints /health e /predict)"
  value       = aws_lambda_function_url.api.function_url
}

output "ecr_repository_url" {
  description = "URL do repositório ECR — usar para fazer push da imagem Docker antes do primeiro deploy"
  value       = aws_ecr_repository.api.repository_url
}
