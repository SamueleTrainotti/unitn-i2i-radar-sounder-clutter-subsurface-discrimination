param(
    [string]$ImageName,
    [string]$TagName
)

# Validate input
if (-not $ImageName -or -not $TagName) {
    Write-Host "Usage: .\launch_docker.ps1 <ImageName> <TagName>"
    exit 1
}

# Set variables
$USER = $env:USERNAME
$USERID = 1000   # Windows does not use UID/GID like Linux; use a default or map as needed
$GROUPID = 1000
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ContainerProjectDir = "/home/containeruser"

# Build the Docker image
docker build `
    --build-arg USERID=$USERID `
    --build-arg GROUPID=$GROUPID `
    --build-arg REPO_DIR=$ContainerProjectDir `
    -t "${USER}/${ImageName}:${TagName}" $ProjectDir

# Run the Docker container
docker run -it -h $ImageName --name "${ImageName}_$USER" `
    --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 `
    -u "${USERID}:${GROUPID}" `
    -v "${ProjectDir}:${ContainerProjectDir}" `
    -v "D:\media:/media" `
    -w "${ContainerProjectDir}" `
    "${USER}/${ImageName}:${TagName}"