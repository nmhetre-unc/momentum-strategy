# infra: the service on AWS

Terraform for running the Compose stack's services on AWS in `us-east-2`.
Nothing here has been applied. `terraform plan` is clean: 15 resources to
add, 0 to change, 0 to destroy.

```
 your IP ──:8000──▶ EC2 t3.small (default VPC, public subnet, public IPv4)
                    ├─ api     container ─┐
                    ├─ worker  container ─┼─▶ RDS PostgreSQL 17, db.t4g.micro
                    └─ one-shots: migrate,│   (private, from the instance's
                       ingest, train ─────┘    security group only)
                    pulls from ECR (instance role, credential helper)
                    reads DATABASE_URL from SSM Parameter Store (SecureString)
                    shell: Session Manager (no SSH, no port 22)
```

| File | What |
|---|---|
| `versions.tf` | Terraform >= 1.11, AWS provider 6.x, random 3.x; default tags |
| `variables.tf` | Region, your IP (`allowed_cidr`), sizes, image tag |
| `network.tf` | Default VPC; two security groups (API from one IP; DB from the instance only) |
| `ecr.tf` | Repository (immutable tags, scan on push, keep 3 images) |
| `database.tf` | RDS, its subnet group, the generated password, the SSM parameter |
| `iam.tf` | The instance role and its inline policy |
| `instance.tf` | The EC2 instance and its cloud-init (Docker, Compose, the `qbt` command) |
| `templates/` | The instance's `compose.yaml` and `/usr/local/bin/qbt` |

## Choices

**Docker Compose on the instance, not one systemd unit per container.**
The repo's `compose.yaml` already encodes how these containers must run:
read-only root filesystem, tmpfs `/tmp`, tini as PID 1, healthchecks, the
worker's 15 s stop grace period (so SIGTERM releases its job), CPU shares
for the API. Systemd units would have to restate every one of those as
`docker run` flags, and drift from the file the tests and the smoke test
exercise. With Compose, the instance's `compose.yaml` is the repo's file
minus the database and minus the migrate dependency. One-shots are
`docker compose run --rm`, the same as locally. A single systemd unit
(`qbt.service`) runs `qbt up` at boot, so a stopped and started instance
comes back.

**Migrations, ingest and training are one-shot commands, never part of API
start-up.** The instance's compose file has no `depends_on: migrate`, and
nothing runs alembic when a container starts. Run them by hand, in order:
`qbt migrate`, `qbt ingest …`, `qbt train …`, then `qbt up`.

**No S3 bucket.** It wouldn't be useful here. `qbt_service.ingest`
downloads from Yahoo and stores each download as an immutable,
content-hashed snapshot in Postgres (`price_series` / `price_bars`).
Backtests and training pin those snapshots, and workers never download.
The raw download is that snapshot, so a copy in S3 would duplicate what the
database already keeps. It would also add a bucket, a policy and write
permissions to the instance role for nothing. If snapshots ever had to
outlive the database, an export to S3 would be the place to add one.

**Secrets.** Terraform generates the database password as an *ephemeral*
value. It is sent to RDS (`password_wo`) and into the SSM SecureString
`/qbt/database-url` (`value_wo`) only through write-only arguments, so it
is in neither the plan nor `terraform.tfstate`. The SSM parameter uses the
AWS managed key `aws/ssm`. Before every command, `qbt` reads the URL into
`/run/qbt/env`, which is on tmpfs and readable by root only. Containers
get it through `env_file`. To rotate it, bump `db_password_version` and
apply, then run `qbt up`.

**Least-privilege instance role** (`iam.tf`): ECR pull on this one
repository, plus the account-wide auth token, which can't be scoped.
`ssm:GetParameter(s)` on `/qbt/*` only. The five actions the SSM Agent needs
for Session Manager. It has no S3, no writes and no wildcard on
parameters, which is why the broader `AmazonSSMManagedInstanceCore` isn't
attached. IMDSv2 is required with a hop limit of 1, so processes in
containers can't reach the instance role at all. No IAM users and no access
keys are created.

**Network.** The default VPC with no NAT gateway. The instance has a
public IPv4 address to reach ECR, SSM and Yahoo over HTTPS. Its security
group allows 8000/tcp from `allowed_cidr` only, and egress on 443 and on
5432 to the database. RDS is not publicly accessible and admits only the
instance's security group. Database connections use TLS with full
verification; see below.

**TLS to RDS: `sslmode=verify-full`.**
1. **Download at boot.** cloud-init downloads AWS's global RDS CA bundle
   ([`global-bundle.pem`](https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem),
   the "any commercial Region" bundle in
   [Using SSL/TLS to encrypt a connection](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/UsingWithRDS.SSL.html)).
   It downloads over HTTPS only (`curl --proto =https --tlsv1.2`), so the
   source, `truststore.pki.rds.amazonaws.com`, is authenticated by the
   host's CA store.
2. **Check the pinned checksum.** The file must match the SHA-256 in
   `rds_ca_bundle_sha256`. AWS publishes no checksum, so that value is the
   hash of the bundle downloaded on 2026-11-15: 114 certificates, all
   self-signed root CAs. AWS asks clients to trust roots only, not
   intermediates. Among them is "Amazon RDS us-east-2 Root CA RSA2048 G1",
   valid to 2061, the CA the database is pinned to
   (`ca_cert_identifier = "rds-ca-rsa2048-g1"`).
3. **Install, or refuse to start.** On a match, the bundle is installed at
   `/opt/qbt/rds-global-bundle.pem` (root-owned, 0644). On a mismatch it is
   discarded, and `qbt` refuses to run anything rather than connect
   unverified.
4. **Mount into every container.** Each container gets the bundle
   read-only at `/etc/ssl/rds/global-bundle.pem`, with `create_host_path:
   false`, so a missing file fails the container instead of mounting an
   empty directory.
5. **Point the URL at it.** `DATABASE_URL` carries
   `sslmode=verify-full&sslrootcert=/etc/ssl/rds/global-bundle.pem`. libpq
   then requires the server certificate to chain to one of those roots and
   to name the endpoint's hostname.

Checked locally with the image as it ships: non-root user, read-only root,
bundle bind-mounted read-only. Against a Postgres with a self-signed
certificate, `verify-full` failed with "certificate verify failed" and
`require` connected. The success path against a real RDS certificate is
only exercised after an apply.

When AWS adds CAs to the bundle, the boot check fails on purpose.
Download the new bundle, inspect it (`openssl` or `keytool -printcert`),
put its SHA-256 in `rds_ca_bundle_sha256`, and apply.

**The API is plain HTTP on port 8000, reachable from one IP address.**
There is no TLS termination and no domain: no load balancer, no
certificate, no Route 53. That is acceptable for this deployment because
it is a short-lived, measured demo:
- The security group admits a single `/32`, yours, so nothing else on the
  internet can connect at all.
- The API has no authentication, sessions or cookies, and carries no
  credentials or personal data. Requests and responses are backtest
  parameters, public SPY prices, and model metadata that is already in
  this repository.
- Adding TLS would mean an ALB (about $16/month plus LCU charges, and
  billed while idle) or a domain and certificate on the instance. That is
  cost and moving parts out of proportion to a session of a few hours.

The consequence is that traffic between your machine and the instance is
readable and alterable by anyone on the network path. Don't send anything
over it that you wouldn't put in this repo. Before any wider use, put TLS
in front: an ALB with an ACM certificate, or Caddy on the instance with a
domain. Add authentication too, since the API has none.

**Smallest sizes that work.**
- *Instance:* t3.small (2 vCPU, 2 GiB, x86_64; the image is built for
  amd64, so not Graviton). Measured locally, the API (2 processes) peaks at
  347 MiB, the idle worker at 348 MiB, ingest at 144 MiB and training at
  141 MiB. Amazon Linux, Docker, the SSM Agent and the worker's per-job
  compute process sit on top of that, which leaves t3.micro's 1 GiB
  without headroom. The API runs 1 process here (`api_processes`); the
  repo's 2 were sized for 4+ workers.
- *CPU credits:* `standard`, so a busy instance is throttled rather than
  billed for surplus credits (`unlimited` is the T3 default).
- *Database:* db.t4g.micro with 20 GB gp3, the gp3 minimum, and storage
  autoscaling off.

## Cost

Prices are on-demand for us-east-2, from the AWS Price List API
(`aws pricing get-products`), on 2026-11-15. "2 h" means apply, use for
two hours, destroy. EC2 bills per second (60 s minimum), and RDS per
second with a 10-minute minimum.

| Resource | Rate | Per month (730 h) | 2-hour session | Bills while idle or stopped? |
|---|---|---|---|---|
| EC2 t3.small (`aws_instance.app`) | $0.0208/h | $15.18 | $0.042 | No when stopped; yes while running, even idle |
| Public IPv4 on the instance | $0.005/h | $3.65 | $0.010 | **Yes while the instance runs**, idle or not. Released on stop (none is reserved: no Elastic IP) |
| EBS root, gp3 12 GB | $0.08/GB-mo | $0.96 | $0.003 | **Yes, even when the instance is stopped**, until it is terminated |
| RDS db.t4g.micro, Single-AZ (`aws_db_instance.db`) | $0.016/h | $11.68 | $0.032 | No when stopped, but **RDS restarts a stopped instance after 7 days** |
| RDS storage, gp3 20 GB | $0.115/GB-mo | $2.30 | $0.006 | **Yes, even when the instance is stopped** |
| RDS backup storage (1-day retention) | $0.095/GB-mo beyond the free allocation | $0 (included up to 20 GB) | $0 | Automated backups are deleted with the instance; no final snapshot is taken (`skip_final_snapshot`), so **no snapshot is left billing** |
| ECR storage (~0.18 GB per image, at most 3) | $0.10/GB-mo | ≤ $0.05 | ~$0 | **Yes**, until the images are deleted (`force_delete` deletes them on destroy) |
| SSM parameter (Standard, SecureString with `aws/ssm`) | no charge | $0 | $0 | No |
| KMS (AWS managed keys `aws/ssm`, `aws/rds`, `aws/ebs`) | no charge | $0 | $0 | No |
| IAM role, policy, instance profile; security groups and rules; DB subnet group; ECR lifecycle policy | no charge | $0 | $0 | No |
| Session Manager | no charge | $0 | $0 | No |
| Data transfer (ECR pulls in-Region, API responses) | in: free; out: first 100 GB/month free | ~$0 | ~$0 | No |
| **Total** | | **≈ $33.8** | **≈ $0.09** | |

**This account** is on the Free plan, with $100 in credits, expiring
2027-05-15, and cannot be charged. Every figure above draws down those
credits: a 2-hour session uses about $0.09 of them, and leaving everything
running uses about $33.80 a month, which would exhaust $100 in about three
months. The Free plan ends when the credits run out, and the account closes
with it.

**Not created, deliberately:** no NAT gateway (~$32.85/month idle plus data
processing), no Elastic IP (billed while unattached too), no load balancer,
no EBS or RDS snapshots, no CloudWatch log groups or alarms, no Secrets
Manager secret ($0.40/month).

**What keeps billing after you stop rather than destroy:**
- the EBS volume and the RDS storage, about $3.26/month;
- ECR storage;
- the RDS instance itself once RDS restarts it after 7 days.

`terraform destroy` removes all of it, including the ECR images and the
automated backups, and takes no final snapshot.

## Free Tier check (accounts created on or after 2025-07-15)

- **Credits, not free quotas.** New accounts get $100 in credits at
  sign-up and can earn up to $100 more. Usage in this table is charged
  against the credits. On the **Free plan** you are never charged, but the
  plan ends after 6 months or when the credits run out, whichever comes
  first. The account then closes, and its resources are deleted unless you
  upgrade within 90 days.
  [Choosing a plan](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/free-tier-plans.html)
- **On the Paid plan,** usage beyond the remaining credits is billed at the
  rates above. [Same page](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/free-tier-plans.html)
- **Service availability for project accounts** (the new sign-up). EC2,
  EBS, ECR, RDS, Systems Manager, VPC, IAM, KMS and STS are all "included
  in the Free Tier", so none needs the Paid plan.
  [Supported services for Sign up for AWS (new)](https://docs.aws.amazon.com/accounts/latest/reference/supported-services-sign-up-new.html)
  The Free-plan service control policy allows `ec2:*`, `ecr:*`, `rds:*`,
  `ssm:*`, `ssmmessages:*`, `ec2messages:*`, `iam:*`, `kms:*` and `sts:*`.
  It denies Spot, Dedicated Hosts, instance-store roots and roles under
  `/managed/`, and nothing here uses those.
  [Managed policies for projects](https://docs.aws.amazon.com/accounts/latest/reference/scps-and-rcps-for-projects.html)
- **EC2:** for accounts created on or after 2025-07-15, the types marked
  Free Tier eligible are t3.micro, t3.small, t4g.micro, t4g.small,
  c7i-flex.large and m7i-flex.large.
  [EC2 Free Tier usage](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-free-tier-usage.html)
  This account's `describe-instance-types --filters free-tier-eligible=true`
  also lists t8i.micro and t8i.small. **t3.small is eligible.**
- **RDS:** "Free plan is available on db.t3.micro and db.t4g.micro instances"
  for MySQL, PostgreSQL, MariaDB and SQL Server Express.
  [Amazon RDS Free Tier](https://aws.amazon.com/rds/free/) **db.t4g.micro
  PostgreSQL is eligible.** A larger class would need the Paid plan.
- **Region:** a project account creates Regional resources only in its
  assigned Region (us-east-2, eu-north-1 or ap-southeast-2).
  [AWS Regions for your projects](https://docs.aws.amazon.com/accounts/latest/reference/project-regions.html)
  That page also says CloudWatch Logs is available in us-east-1, which is
  one more reason container logs stay on the instance: json-file, 3 × 10 MB.

## Runbook

Prerequisites: AWS CLI signed in (`aws login`), Docker, Terraform >= 1.11,
and the [Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html)
for `aws ssm start-session`.

```bash
cd infra
printf 'allowed_cidr = "%s/32"\n' "$(curl -s https://checkip.amazonaws.com)" > terraform.tfvars
terraform init
terraform plan -out=qbt.tfplan
terraform apply qbt.tfplan          # ~5-10 min, mostly RDS

# Build and push the image under the tag the instance expects (image_tag, default v1).
REPO=$(terraform output -raw ecr_repository_url)
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin "${REPO%%/*}"
docker build --platform linux/amd64 -t "$REPO:v1" ..
docker push "$REPO:v1"

# On the instance.
aws ssm start-session --region us-east-2 --target "$(terraform output -raw instance_id)"
sudo qbt migrate
sudo qbt ingest SPY --start 2008-01-01 --end 2025-01-01
sudo qbt train  SPY --start 2008-01-01 --end 2025-01-01
sudo qbt up

# From your machine.
curl -s "$(terraform output -raw api_url)/health"
curl -s -X POST "$(terraform output -raw api_url)/predict"

# Done: remove everything (images, database, instance, parameter).
terraform destroy
```

A new build: push it under a new tag, set `image_tag`, and apply. The
instance is replaced, because its compose file is in user data. Then run
`qbt migrate` and `qbt up` on the new instance; the data stays in RDS.

The state (`terraform.tfstate`), plans and `terraform.tfvars` (your IP)
are gitignored. `.terraform.lock.hcl` is committed.
