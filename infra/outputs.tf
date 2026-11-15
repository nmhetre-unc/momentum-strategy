output "ecr_repository_url" {
  description = "Push the image here (see infra/README.md)."
  value       = aws_ecr_repository.service.repository_url
}

output "image" {
  description = "The image the instance runs (image_tag)."
  value       = local.image
}

output "instance_id" {
  description = "aws ssm start-session --target <this>"
  value       = aws_instance.app.id
}

output "api_url" {
  description = "Reachable from allowed_cidr only. The public IP changes if the instance is stopped and started."
  value       = "http://${aws_instance.app.public_ip}:8000"
}

output "db_endpoint" {
  description = "RDS endpoint (private: reachable from the instance only)."
  value       = aws_db_instance.db.endpoint
}

output "database_url_parameter" {
  description = "SSM SecureString holding DATABASE_URL."
  value       = aws_ssm_parameter.database_url.name
}
