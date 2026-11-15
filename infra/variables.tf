variable "region" {
  description = "AWS Region. A project account from the new sign-up can only create Regional resources in its assigned Region."
  type        = string
  default     = "us-east-2"
}

variable "name" {
  description = "Prefix for resource names, and the Project tag."
  type        = string
  default     = "qbt"
}

variable "allowed_cidr" {
  description = "The only source allowed to reach the API on port 8000: your public IP as a /32 (curl -s https://checkip.amazonaws.com). Set it in terraform.tfvars, which is gitignored."
  type        = string

  validation {
    condition     = can(cidrhost(var.allowed_cidr, 0)) && endswith(var.allowed_cidr, "/32")
    error_message = "allowed_cidr must be a single IPv4 address as a /32, e.g. 203.0.113.7/32."
  }
}

variable "instance_type" {
  description = "EC2 instance for the api and worker containers. Free Tier eligible for accounts created after 2025-07-15: t3.micro, t3.small, t4g.*, t8i.*, c7i-flex.large, m7i-flex.large. The image is built for x86_64, so not t4g."
  type        = string
  default     = "t3.small"
}

variable "root_volume_gb" {
  description = "Root EBS volume (gp3). Holds the OS, Docker and the ~0.75 GB image."
  type        = number
  default     = 12
}

variable "db_instance_class" {
  description = "RDS instance class. The Free plan covers db.t3.micro and db.t4g.micro."
  type        = string
  default     = "db.t4g.micro"
}

variable "db_allocated_storage_gb" {
  description = "RDS storage (gp3). 20 GB is the minimum for PostgreSQL on gp3."
  type        = number
  default     = 20
}

variable "db_password_version" {
  description = "Bump to generate a new database password and write it to RDS and to the SSM parameter in the same apply. The password is never stored in state."
  type        = number
  default     = 1
}

variable "image_tag" {
  description = "Tag of the qbt-service image in ECR that the instance runs. Tags are immutable in this repository, so deploy a new build by pushing a new tag and changing this."
  type        = string
  default     = "v1"
}

variable "api_processes" {
  description = "uvicorn processes in the api container (WEB_CONCURRENCY). One on a 2-vCPU instance with one worker; the repo default of 2 was sized for 4+ workers (bench/results/service/get.json)."
  type        = number
  default     = 1
}

variable "compose_version" {
  description = "Docker Compose plugin release installed on the instance, verified by SHA-256."
  type        = string
  default     = "v5.6.1"
}

variable "compose_sha256" {
  description = "SHA-256 of docker-compose-linux-x86_64 for compose_version (from the release's .sha256 file)."
  type        = string
  default     = "47737f4236e6d3ccea7291995203f8f018058783b4c3a9469119811a0b5db8c0"
}
