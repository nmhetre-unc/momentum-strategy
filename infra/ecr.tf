resource "aws_ecr_repository" "service" {
  name = "${var.name}-service"

  # A tag always means the same image: deploy a new build under a new tag.
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true # basic scanning, no charge
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  # `terraform destroy` deletes the images too, so nothing is left billing
  # $0.10/GB-month for storage.
  force_delete = true
}

# Storage is the only ECR cost: keep the 3 newest images.
resource "aws_ecr_lifecycle_policy" "service" {
  repository = aws_ecr_repository.service.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep the 3 newest images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 3
      }
      action = { type = "expire" }
    }]
  })
}
