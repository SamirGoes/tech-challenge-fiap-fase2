variable "aws_region" {
  description = "Região AWS onde os recursos serão criados"
  type        = string
  default     = "us-east-1"
}

variable "nome_projeto" {
  description = "Prefixo usado no nome dos recursos"
  type        = string
  default     = "tech-challenge-fiap-fase2-ga"
}

variable "anthropic_api_key" {
  description = "Chave da API Anthropic. Guardada no SSM Parameter Store (SecureString), nunca no código. Passar via TF_VAR_anthropic_api_key ou terraform.tfvars (não versionado)."
  type        = string
  sensitive   = true
}