# Terraform 1.11+ for write-only arguments (password_wo, value_wo) and
# ephemeral resources: the database password is generated, sent to RDS and
# to SSM Parameter Store, and never written to the state file or the plan.
terraform {
  required_version = ">= 1.11"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.7"
    }
  }

  # Local state (infra/terraform.tfstate), gitignored. It holds resource IDs
  # and endpoints, not the database password (see above).
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = var.name
      ManagedBy = "terraform"
      Source    = "momentum-strategy/infra"
    }
  }
}
