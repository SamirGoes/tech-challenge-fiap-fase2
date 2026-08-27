output "function_url" {
  description = "URL pública HTTPS da API (endpoints /health e /predict)"
  value       = aws_apigatewayv2_stage.default.invoke_url
}

output "ecr_repository_url" {
  description = "URL do repositório ECR — usar para fazer push da imagem Docker antes do primeiro deploy"
  value       = aws_ecr_repository.api.repository_url
}

output "frontend_bucket" {
  description = "Nome do bucket S3 — usar para fazer upload do build do Angular (ng build)"
  value       = aws_s3_bucket.frontend.id
}

output "frontend_url" {
  description = "URL pública HTTPS do front (CloudFront) — serve o Angular e encaminha /predict, /health e /status pro API Gateway no mesmo domínio"
  value       = "https://${aws_cloudfront_distribution.app.domain_name}"
}

output "cloudfront_distribution_id" {
  description = "ID da distribuição CloudFront — usar para invalidar o cache após cada novo deploy do front"
  value       = aws_cloudfront_distribution.app.id
}
