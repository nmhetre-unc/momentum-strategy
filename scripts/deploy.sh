#!/usr/bin/env bash
# Deploy one published image to the running AWS stack (infra/).
#
#   scripts/deploy.sh <commit-sha> [--migrate]
#
# Run by a person, with the short-lived credentials from `aws login`; no
# AWS keys are stored anywhere. Nothing in GitHub can reach AWS: the
# account's SCP denies the GitHub OIDC provider (README, "Deploying to AWS").
#
# 1. Checks the AWS credentials, finds the one running instance
#    (tag Name=qbt-app) and its SSM agent, and the ECR repository.
# 2. Finds the successful .github/workflows/publish.yml run on main for the
#    commit and downloads the digest it recorded (artifact image-digest).
# 3. Resolves ghcr.io/nmhetre-unc/momentum-strategy:<sha> and refuses unless
#    GHCR serves exactly that digest.
# 4. Pulls that digest from GHCR, pushes it to ECR as qbt-service:<sha>, and
#    checks ECR now holds the same digest. (Not `docker buildx imagetools
#    create`: given a single-platform manifest, as publish.yml produces, it
#    wraps it in a new manifest list with a different digest. D3b caught
#    this on its first run.)
# 5. On the instance, over SSM Run Command: pulls qbt-service@<digest> from
#    ECR (the instance never pulls from anywhere else), tags it with the
#    image name its compose file runs (see "Why retag" below), optionally
#    `qbt migrate`, then `qbt up`, and waits for GET /health = 200 and for
#    the api and worker containers to run the new image.
#
# Why retag: the instance's compose file names one tag, infra/'s image_tag
# (default v1), and ECR tags are immutable. Rather than change infra/ or
# rewrite that file, the pulled digest is tagged locally under that name;
# `qbt up` (and qbt.service at boot) then runs it without pulling.
#
# Every line of output passes through bench/scrub.py.
#
# Needs: aws CLI v2 (signed in), docker with buildx, gh (signed in), python3.
set -euo pipefail

REGION=us-east-2
GH_REPO=nmhetre-unc/momentum-strategy
GHCR=ghcr.io/$GH_REPO
ECR_REPOSITORY=qbt-service
INSTANCE_NAME=qbt-app
HEALTH_TIMEOUT_SECONDS=180

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# All output, stdout and stderr, through the scrubber.
exec > >(python3 -u "$repo_root/bench/scrub.py") 2>&1
scrub_pid=$!

work=""
cleanup() {
    local status=$?
    # The temporary Docker config holds registry logins: never leave it.
    if [[ -n "$work" && -d "$work" ]]; then
        rm -rf -- "$work"
    fi
    exec >&- 2>&-
    wait "$scrub_pid" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT

# UTC timestamps on every message, so each phase can be timed from the output.
now() { date -u +%Y-%m-%dT%H:%M:%S.%3NZ; }
die() { echo "$(now) deploy: FAILED: $*" >&2; exit 1; }
say() { echo "$(now) deploy: $*"; }

usage() { echo "usage: scripts/deploy.sh <40-character commit sha> [--migrate]" >&2; exit 2; }

sha="${1:-}"
[[ "$sha" =~ ^[0-9a-f]{40}$ ]] || usage
migrate=no
case "${2:-}" in
    "") ;;
    --migrate) migrate=yes ;;
    *) usage ;;
esac

for tool in aws docker gh python3; do
    command -v "$tool" >/dev/null || die "$tool is not installed"
done
docker buildx version >/dev/null 2>&1 || die "docker buildx is not available"

json_field() {  # json_field <python expression on d> < json
    python3 -c 'import json, sys; d = json.load(sys.stdin); print(eval(sys.argv[1]))' "$1"
}

# ---- 1. AWS: credentials, instance, ECR -------------------------------------

if ! identity=$(aws sts get-caller-identity --region "$REGION" --output json 2>&1); then
    die "AWS credentials are missing or expired. Run \`aws login\` and try again.
$identity"
fi
account=$(json_field 'd["Account"]' <<< "$identity")
say "AWS account $account as $(json_field 'd["Arn"]' <<< "$identity")"

instances=$(aws ec2 describe-instances --region "$REGION" \
    --filters "Name=tag:Name,Values=$INSTANCE_NAME" "Name=instance-state-name,Values=running" \
    --query 'Reservations[].Instances[].InstanceId' --output text)
read -r -a instance_ids <<< "$instances"
case "${#instance_ids[@]}" in
    0) die "no running instance tagged Name=$INSTANCE_NAME in $REGION. The stack is not up:
apply infra/ first (infra/README.md, Runbook), then run this again." ;;
    1) instance=${instance_ids[0]} ;;
    *) die "more than one running instance tagged Name=$INSTANCE_NAME: ${instance_ids[*]}" ;;
esac
ping=$(aws ssm describe-instance-information --region "$REGION" \
    --filters "Key=InstanceIds,Values=$instance" \
    --query 'InstanceInformationList[0].PingStatus' --output text)
[[ "$ping" == "Online" ]] || die "instance $instance is running but its SSM agent is '$ping', not Online (still booting?)"
say "instance $instance, SSM Online"

ecr_uri=$(aws ecr describe-repositories --region "$REGION" --repository-names "$ECR_REPOSITORY" \
    --query 'repositories[0].repositoryUri' --output text 2>/dev/null) \
    || die "ECR repository $ECR_REPOSITORY not found in $REGION (apply infra/ first)"
ecr_registry=${ecr_uri%%/*}

# ---- 2. The digest the publish workflow recorded ----------------------------

gh auth status --hostname github.com >/dev/null 2>&1 || die "gh is not signed in: run \`gh auth login\`"
run_id=$(gh run list --repo "$GH_REPO" --workflow publish.yml --branch main --commit "$sha" \
    --status success --limit 1 --json databaseId --jq '.[0].databaseId // empty')
[[ -n "$run_id" ]] || die "no successful publish.yml run on main for $sha
(https://github.com/$GH_REPO/actions/workflows/publish.yml). Only images built from main deploy."

work=$(mktemp -d)
export DOCKER_CONFIG="$work/docker"   # registry logins live here, deleted on exit
mkdir -m 0700 "$DOCKER_CONFIG"

gh run download "$run_id" --repo "$GH_REPO" --name image-digest --dir "$work/artifact" \
    || die "could not download the image-digest artifact of run $run_id (expired after 90 days?)"
published=$(tr -d '[:space:]' < "$work/artifact/digest.txt")
[[ "$published" =~ ^sha256:[0-9a-f]{64}$ ]] || die "run $run_id recorded no valid digest: '$published'"
say "publish run $run_id recorded $GHCR:$sha = $published"

# ---- 3. GHCR serves that digest? ---------------------------------------------

manifest_digest() {  # manifest_digest <ref>: the digest a registry serves for <ref>
    docker buildx imagetools inspect "$1" --format '{{json .Manifest}}' | json_field 'd["digest"]'
}

if ! served=$(manifest_digest "$GHCR:$sha" 2>/dev/null); then
    # A private package: log in with the gh token (needs read:packages).
    gh auth token | docker login ghcr.io --username "$(gh api user --jq .login)" --password-stdin \
        >/dev/null 2>&1 || true
    served=$(manifest_digest "$GHCR:$sha" 2>/dev/null) \
        || die "cannot read $GHCR:$sha. If the package is private: gh auth refresh -s read:packages"
fi
[[ "$served" == "$published" ]] || die "digest mismatch for $GHCR:$sha:
  GHCR serves        $served
  publish run $run_id recorded $published
Refusing to deploy an image the workflow did not publish."
say "GHCR serves the recorded digest"

# ---- 4. Copy into ECR, by digest ---------------------------------------------

aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$ecr_registry" >/dev/null \
    || die "docker login to ECR failed"

ecr_digest() {
    aws ecr describe-images --region "$REGION" --repository-name "$ECR_REPOSITORY" \
        --image-ids "imageTag=$sha" --query 'imageDetails[0].imageDigest' --output text 2>/dev/null
}
if existing=$(ecr_digest); then
    [[ "$existing" == "$published" ]] \
        || die "ECR already has $ECR_REPOSITORY:$sha with a different digest ($existing); tags are
immutable there. Remove it, then run this again:
  aws ecr batch-delete-image --region $REGION --repository-name $ECR_REPOSITORY --image-ids imageTag=$sha"
    say "ECR already has $ECR_REPOSITORY:$sha at that digest"
else
    say "pulling $GHCR@$published"
    docker pull --quiet --platform linux/amd64 "$GHCR@$published"
    docker tag "$GHCR@$published" "$ecr_uri:$sha"
    say "pushing $ecr_uri:$sha"
    docker push --quiet "$ecr_uri:$sha"
    docker image rm "$ecr_uri:$sha" >/dev/null   # the local tag only; the pulled image stays
    in_ecr=$(ecr_digest) || die "the copy to ECR did not land"
    [[ "$in_ecr" == "$published" ]] || die "ECR holds $in_ecr after the copy, not $published"
    say "ECR holds $ECR_REPOSITORY:$sha at the same digest"
fi

# ---- 5. Redeploy on the instance, over SSM -----------------------------------

remote_script=$(cat <<EOF
set -euo pipefail
stamp() { echo "\$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ) instance: \$*"; }
ref='$ecr_uri@$published'
target=\$(sed -n 's/^  image: //p' /opt/qbt/compose.yaml | head -n 1)
[ -n "\$target" ] || { echo "no image line in /opt/qbt/compose.yaml" >&2; exit 1; }
stamp "pull \$ref"
docker pull --quiet "\$ref"
docker tag "\$ref" "\$target"
stamp "tagged \$ref as \$target"
if [ "$migrate" = yes ]; then /usr/local/bin/qbt migrate; stamp "migrated"; fi
/usr/local/bin/qbt up
stamp "qbt up done"
healthy=no
for _ in \$(seq $((HEALTH_TIMEOUT_SECONDS / 2))); do
  if curl -fsS --max-time 3 http://127.0.0.1:8000/health; then healthy=yes; echo; break; fi
  sleep 2
done
[ "\$healthy" = yes ] || { echo "GET /health did not answer 200 within $HEALTH_TIMEOUT_SECONDS s" >&2; /usr/local/bin/qbt ps; exit 1; }
want=\$(docker image inspect --format '{{.Id}}' "\$ref")
for service in api worker; do
  for c in \$(docker compose -p qbt -f /opt/qbt/compose.yaml ps -q "\$service"); do
    have=\$(docker inspect --format '{{.Image}}' "\$c")
    [ "\$have" = "\$want" ] || { echo "\$service container \$c runs \$have, not \$want" >&2; exit 1; }
  done
done
stamp "healthy; api and worker run \$ref"
/usr/local/bin/qbt ps
EOF
)

params=$(python3 -c '
import base64, json, sys
script = base64.b64encode(sys.argv[1].encode()).decode()
print(json.dumps({"commands": [f"echo {script} | base64 -d > /tmp/qbt-deploy.sh",
                               "bash /tmp/qbt-deploy.sh"],
                  "executionTimeout": ["900"]}))' "$remote_script")

say "redeploying on $instance over SSM (pull, tag${migrate/yes/, migrate}, qbt up)"
command_id=$(aws ssm send-command --region "$REGION" --instance-ids "$instance" \
    --document-name AWS-RunShellScript --parameters "$params" --timeout-seconds 60 \
    --comment "scripts/deploy.sh $sha" --query 'Command.CommandId' --output text)

status=Pending
deadline=$((SECONDS + 960))
while (( SECONDS < deadline )); do
    sleep 3
    status=$(aws ssm get-command-invocation --region "$REGION" --command-id "$command_id" \
        --instance-id "$instance" --query Status --output text 2>/dev/null) || continue
    case "$status" in Pending|InProgress|Delayed) continue ;; *) break ;; esac
done
invocation=$(aws ssm get-command-invocation --region "$REGION" --command-id "$command_id" \
    --instance-id "$instance" --output json)
echo "---- on the instance (stdout)"
json_field 'd["StandardOutputContent"]' <<< "$invocation"
echo "---- on the instance (stderr)"
json_field 'd["StandardErrorContent"]' <<< "$invocation"
[[ "$status" == "Success" ]] || die "the redeploy on $instance ended '$status' (SSM command $command_id)"

public_ip=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$instance" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
say "deployed $sha ($published) to $instance; API http://$public_ip:8000 (from allowed_cidr only)"
