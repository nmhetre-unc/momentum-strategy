# The instance role: pull this one ECR repository, read this project's SSM
# parameters, and register with Session Manager. Nothing else: no S3, no
# write access to anything, no long-lived keys. The containers can't use it
# either: IMDSv2 with a hop limit of 1 (ec2.tf) keeps instance credentials
# away from processes inside Docker's bridge network.

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

locals {
  parameter_arn_prefix = "arn:${data.aws_partition.current.partition}:ssm:${var.region}:${data.aws_caller_identity.current.account_id}:parameter/${var.name}"
}

data "aws_iam_policy_document" "instance_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
    # No aws:SourceAccount condition: EC2 doesn't document setting it when it
    # assumes an instance role, and a condition it doesn't satisfy would
    # leave the instance without credentials. Only this account can attach
    # the role to an instance (iam:PassRole).
  }
}

resource "aws_iam_role" "instance" {
  name               = "${var.name}-instance"
  path               = "/${var.name}/"
  description        = "qbt EC2 instance: pull the qbt image, read /${var.name}/* parameters, Session Manager"
  assume_role_policy = data.aws_iam_policy_document.instance_trust.json
}

data "aws_iam_policy_document" "instance" {
  # ECR: an auth token can't be scoped to a repository; pulling can.
  statement {
    sid       = "EcrAuthToken"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    sid = "EcrPullThisRepository"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
    ]
    resources = [aws_ecr_repository.service.arn]
  }

  # SSM Parameter Store: this project's parameters only, read only. The
  # SecureString is encrypted with the AWS managed key aws/ssm, whose key
  # policy already lets account principals decrypt through SSM, so no
  # kms:Decrypt statement is needed.
  statement {
    sid       = "ReadProjectParameters"
    actions   = ["ssm:GetParameter", "ssm:GetParameters"]
    resources = ["${local.parameter_arn_prefix}/*"]
  }

  # Session Manager (shell access without SSH or an open port): the minimum
  # the SSM Agent needs to register and open sessions. AWS's broader managed
  # policy (AmazonSSMManagedInstanceCore) also grants ssm:GetParameter on
  # every parameter, which is why it isn't attached.
  statement {
    sid = "SessionManager"
    actions = [
      "ssm:UpdateInstanceInformation",
      "ssmmessages:CreateControlChannel",
      "ssmmessages:CreateDataChannel",
      "ssmmessages:OpenControlChannel",
      "ssmmessages:OpenDataChannel",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "instance" {
  name   = "${var.name}-instance"
  role   = aws_iam_role.instance.id
  policy = data.aws_iam_policy_document.instance.json
}

resource "aws_iam_instance_profile" "instance" {
  name = "${var.name}-instance"
  path = "/${var.name}/"
  role = aws_iam_role.instance.name
}
