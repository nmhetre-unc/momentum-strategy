# One EC2 instance running the api and worker containers with Docker
# Compose (see infra/README.md for why Compose rather than systemd units
# per container). No SSH key and no port 22: shell access is through
# Session Manager.

data "aws_ssm_parameter" "al2023" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

locals {
  # The RDS CA bundle: downloaded and verified at boot on the host, mounted
  # read-only into every container at the path DATABASE_URL names.
  rds_ca_url            = "https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem"
  rds_ca_host_path      = "/opt/qbt/rds-global-bundle.pem"
  rds_ca_container_path = "/etc/ssl/rds/global-bundle.pem"

  image = "${aws_ecr_repository.service.repository_url}:${var.image_tag}"

  compose_yaml = templatefile("${path.module}/templates/compose.yaml.tftpl", {
    image                 = local.image
    api_processes         = var.api_processes
    rds_ca_host_path      = local.rds_ca_host_path
    rds_ca_container_path = local.rds_ca_container_path
  })

  registry = split("/", aws_ecr_repository.service.repository_url)[0]

  cloud_config = {
    write_files = [
      { path = "/opt/qbt/compose.yaml", permissions = "0644", content = local.compose_yaml },
      {
        path        = "/opt/qbt/qbt.conf"
        permissions = "0644"
        content     = "QBT_REGION=${var.region}\nQBT_PARAMETER=${aws_ssm_parameter.database_url.name}\nQBT_RDS_CA=${local.rds_ca_host_path}\n"
      },
      { path = "/usr/local/bin/qbt", permissions = "0755", content = file("${path.module}/templates/qbt.sh") },
      {
        # Docker pulls from ECR with the instance role's credentials through
        # the credential helper: no `docker login`, no stored password.
        path        = "/root/.docker/config.json"
        permissions = "0600"
        content     = jsonencode({ credHelpers = { (local.registry) = "ecr-login" } })
      },
      {
        # Starts api and worker on boot once they have been deployed;
        # `qbt up` the first time, after `qbt migrate`.
        path        = "/etc/systemd/system/qbt.service"
        permissions = "0644"
        content     = <<-EOT
          [Unit]
          Description=qbt api and worker (docker compose)
          Requires=docker.service
          After=docker.service network-online.target
          Wants=network-online.target

          [Service]
          Type=oneshot
          RemainAfterExit=yes
          ExecStart=/usr/local/bin/qbt up
          ExecStop=/usr/local/bin/qbt down
          TimeoutStartSec=300

          [Install]
          WantedBy=multi-user.target
        EOT
      },
    ]
    runcmd = [
      # RDS CA bundle: HTTPS only (TLS 1.2+, the host's own CA store
      # authenticates truststore.pki.rds.amazonaws.com), then the pinned
      # SHA-256. On a mismatch the file is removed, and `qbt` refuses to
      # start anything, rather than connecting without verification.
      ["curl", "--proto", "=https", "--tlsv1.2", "-fsSL", "-o", "${local.rds_ca_host_path}.new", local.rds_ca_url],
      ["sh", "-c", "echo '${var.rds_ca_bundle_sha256}  ${local.rds_ca_host_path}.new' | sha256sum -c - && install -m 0644 -o root -g root ${local.rds_ca_host_path}.new ${local.rds_ca_host_path}; rm -f ${local.rds_ca_host_path}.new"],
      ["dnf", "install", "-y", "docker", "amazon-ecr-credential-helper"],
      ["systemctl", "enable", "--now", "docker"],
      ["mkdir", "-p", "/usr/local/lib/docker/cli-plugins"],
      ["curl", "-fsSL", "-o", "/usr/local/lib/docker/cli-plugins/docker-compose",
      "https://github.com/docker/compose/releases/download/${var.compose_version}/docker-compose-linux-x86_64"],
      ["sh", "-c", "echo '${var.compose_sha256}  /usr/local/lib/docker/cli-plugins/docker-compose' | sha256sum -c - || { rm -f /usr/local/lib/docker/cli-plugins/docker-compose; exit 1; }"],
      ["chmod", "0755", "/usr/local/lib/docker/cli-plugins/docker-compose"],
      ["systemctl", "daemon-reload"],
      ["systemctl", "enable", "qbt.service"],
    ]
  }
}

resource "aws_instance" "app" {
  ami                         = data.aws_ssm_parameter.al2023.insecure_value
  instance_type               = var.instance_type
  subnet_id                   = data.aws_subnets.default.ids[0]
  vpc_security_group_ids      = [aws_security_group.app.id]
  iam_instance_profile        = aws_iam_instance_profile.instance.name
  associate_public_ip_address = true # billed $0.005/h while running; released when stopped

  # Burstable credits: "standard" can't bill for surplus CPU, unlike the
  # T3 default "unlimited". Sustained load is throttled to the baseline instead.
  credit_specification {
    cpu_credits = "standard"
  }

  # IMDSv2 only, hop limit 1: containers (one hop further, behind Docker's
  # bridge) can't reach instance metadata, so they can't use the role.
  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
    http_endpoint               = "enabled"
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_gb
    encrypted             = true
    delete_on_termination = true
  }

  user_data                   = "#cloud-config\n${yamlencode(local.cloud_config)}"
  user_data_replace_on_change = true

  tags = { Name = "${var.name}-app" }

  lifecycle {
    # A newer Amazon Linux AMI shouldn't replace a running instance on the
    # next plan; replace it deliberately (terraform apply -replace=...).
    ignore_changes = [ami]
  }
}
