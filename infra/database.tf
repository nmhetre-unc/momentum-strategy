# RDS PostgreSQL 17, the smallest Free plan class, single AZ, not public.

# Ephemeral: generated during plan/apply and never stored in state. It is
# sent only through write-only arguments (password_wo, value_wo), which RDS
# and SSM receive when db_password_version changes. Letters and digits only,
# so it needs no escaping inside the database URL.
ephemeral "random_password" "db" {
  length  = 40
  special = false
}

resource "aws_db_subnet_group" "db" {
  name       = "${var.name}-db"
  subnet_ids = data.aws_subnets.default.ids
}

resource "aws_db_instance" "db" {
  identifier     = "${var.name}-db"
  engine         = "postgres"
  engine_version = "17" # latest 17.x; minor upgrades applied in the maintenance window
  instance_class = var.db_instance_class

  allocated_storage     = var.db_allocated_storage_gb
  max_allocated_storage = 0 # no autoscaling: storage bills even when stopped
  storage_type          = "gp3"
  storage_encrypted     = true # AWS managed key (aws/rds), no KMS charge

  db_name                = "qbt"
  username               = "qbt"
  password_wo            = ephemeral.random_password.db.result
  password_wo_version    = var.db_password_version
  db_subnet_group_name   = aws_db_subnet_group.db.name
  vpc_security_group_ids = [aws_security_group.db.id]
  publicly_accessible    = false
  multi_az               = false
  port                   = 5432

  # PostgreSQL 15+ on RDS defaults to rds.force_ssl = 1: the URL below uses
  # sslmode=require.

  # Backups: 1 day. Backup storage up to the provisioned size (20 GB) is
  # included, so this costs nothing extra while the instance exists.
  backup_retention_period = 1
  # No final snapshot on destroy: a snapshot would keep billing after the
  # instance is gone. Export the data first if you want it.
  skip_final_snapshot      = true
  delete_automated_backups = true
  deletion_protection      = false

  performance_insights_enabled = false
  monitoring_interval          = 0
  auto_minor_version_upgrade   = true
  apply_immediately            = true
}

# The database URL the containers read (DATABASE_URL), as a SecureString
# encrypted with the AWS managed key aws/ssm (no KMS charge). Standard
# parameters have no charge. Write-only, so the URL with its password is
# not in state either.
resource "aws_ssm_parameter" "database_url" {
  name        = "/${var.name}/database-url"
  description = "DATABASE_URL for the qbt api, worker and one-shot commands"
  type        = "SecureString"
  tier        = "Standard"
  value_wo = format(
    "postgresql+psycopg://%s:%s@%s/%s?sslmode=require",
    aws_db_instance.db.username,
    ephemeral.random_password.db.result,
    aws_db_instance.db.endpoint,
    aws_db_instance.db.db_name,
  )
  value_wo_version = var.db_password_version
}
