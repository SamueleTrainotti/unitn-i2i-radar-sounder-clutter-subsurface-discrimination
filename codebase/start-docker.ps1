# Path to Docker Desktop
$dockerDesktopPath = "C:\Program Files\Docker\Docker\Docker Desktop.exe"

# Check if Docker Desktop is running
$dockerRunning = Get-Process -Name "Docker Desktop" -ErrorAction SilentlyContinue
if (-not $dockerRunning) {
    Write-Host "Starting Docker Desktop..."
    Start-Process "$dockerDesktopPath"
} else {
    Write-Host "Docker Desktop already running."
}

# Wait for Docker to become ready
Write-Host "Waiting for Docker to be ready..."
while ($true) {
    docker info >$null 2>&1
    if ($LASTEXITCODE -eq 0) {
        break
    }
    Start-Sleep -Seconds 2
}
Write-Host "Docker is ready!"

# Start your container (example: my-container)
$containerName = "samueletrainotti_samut"

# Check if container exists
$containerExists = docker ps -a --format "{{.Names}}" | Where-Object { $_ -eq $containerName }
if ($containerExists) {
    # Start container if it’s not running
    $containerRunning = docker ps --format "{{.Names}}" | Where-Object { $_ -eq $containerName }
    if (-not $containerRunning) {
        Write-Host "Starting container: $containerName"
        docker start $containerName
    } else {
        Write-Host "Container '$containerName' is already running."
    }
} else {
    Write-Host "Container '$containerName' not found."
}
