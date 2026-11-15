# The account's default VPC: public subnets in three AZs, an internet
# gateway, no NAT gateway. The instance gets a public IPv4 address so it can
# reach ECR, SSM and Yahoo Finance directly (a NAT gateway would cost about
# $33/month even idle). RDS sits in the same subnets but is not publicly
# accessible and only accepts the instance's security group.

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

# ---- the instance ----

resource "aws_security_group" "app" {
  name        = "${var.name}-app"
  description = "qbt api/worker instance: API from one IP only; no SSH (Session Manager instead)"
  vpc_id      = data.aws_vpc.default.id
}

resource "aws_vpc_security_group_ingress_rule" "api_from_allowed_ip" {
  security_group_id = aws_security_group.app.id
  description       = "API, from allowed_cidr only"
  cidr_ipv4         = var.allowed_cidr
  ip_protocol       = "tcp"
  from_port         = 8000
  to_port           = 8000
}

# Egress: HTTPS (ECR, S3 layer downloads, SSM, the OS package repos,
# Yahoo Finance for ingest) and PostgreSQL to the database. DNS to the VPC
# resolver and NTP to the link-local time service are not filtered by
# security groups.
resource "aws_vpc_security_group_egress_rule" "https" {
  security_group_id = aws_security_group.app.id
  description       = "HTTPS out"
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "tcp"
  from_port         = 443
  to_port           = 443
}

resource "aws_vpc_security_group_egress_rule" "postgres" {
  security_group_id            = aws_security_group.app.id
  description                  = "PostgreSQL to the database"
  referenced_security_group_id = aws_security_group.db.id
  ip_protocol                  = "tcp"
  from_port                    = 5432
  to_port                      = 5432
}

# ---- the database ----

resource "aws_security_group" "db" {
  name        = "${var.name}-db"
  description = "qbt RDS PostgreSQL: from the app instance only"
  vpc_id      = data.aws_vpc.default.id
  # No egress rules: Terraform removes the default allow-all, and a
  # security group is stateful, so replies need none.
}

resource "aws_vpc_security_group_ingress_rule" "db_from_app" {
  security_group_id            = aws_security_group.db.id
  description                  = "PostgreSQL from the app instance"
  referenced_security_group_id = aws_security_group.app.id
  ip_protocol                  = "tcp"
  from_port                    = 5432
  to_port                      = 5432
}
