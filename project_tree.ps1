function Show-Tree([string]$Path = ".", [int]$Depth = 0, [bool]$IsSub = $false) {
    $Tab = "│   " * $Depth
    if ($IsSub) { $Tab = $Tab.Substring(0, $Tab.Length - 1) + "├── " }
    else       { $Tab = $Tab + "├── " }

    # List of ignored items
    $ignoredItems = @(
        ".git", ".gitattributes", ".gitignore",
        ".pytest_cache", "__pycache__", ".vscode",
        "logs", "misc"
    )

    # Order of file types for sorting
    $typeOrder = @(".py", ".sh", ".ipynb", ".toml", ".txt", ".md")

    $Children = Get-ChildItem $Path | Sort-Object @{
        Expression = {
            if ($_.PSIsContainer) { 0 }
            else {
                   $idx = $typeOrder.IndexOf($_.Extension)
                if ($idx -eq -1) { 999 } else { $idx + 1 }
            }
        };
        Ascending = $true
    }

    foreach ($item in $Children) {
        $fullPath = Join-Path $Path $item.Name

        if ($ignoredItems -contains $item.Name) { continue }

        if ($item.PSIsContainer) {
            Write-Output "$Tab$item/"

            # Check if this folder contains only .csv or .log (recursively)
            $nonCsvLogInside = Get-ChildItem $fullPath -Recurse | Where-Object {
                $_.Extension -notin @(".csv", ".log") -and
                $_.Name -notin $ignoredItems
            } | Select-Object -First 1

            if (-not $nonCsvLogInside) {
                # Only .csv or .log inside -> show placeholder
                $newTab = "│   " * ($Depth + 1)
                $hasCsv = Get-ChildItem $fullPath -Recurse | Where-Object { $_.Extension -eq ".csv" } | Select-Object -First 1
                $hasLog = Get-ChildItem $fullPath -Recurse | Where-Object { $_.Extension -eq ".log" } | Select-Object -First 1

                if ($hasCsv -and $hasLog) {
                    Write-Output "$newTab└── (*.csv, *.log)"
                } elseif ($hasCsv) {
                    Write-Output "$newTab└── (*.csv)"
                } elseif ($hasLog) {
                    Write-Output "$newTab└── (*.log)"
                }
            } else {
                # There are other files -> show sub-tree
                Show-Tree $fullPath ($Depth + 1) $true | Out-String -Stream | Where-Object { $_ }
            }
        } else {
            # Show file with specific extensions 
            $allowedExtensions = @(".py", ".ipynb", ".sh", ".toml", ".txt", ".md")
            if ($allowedExtensions -contains $item.Extension) {
                Write-Output "$Tab$item"
            }
        }
    }
}

Write-Output "Project Structure:"
Show-Tree