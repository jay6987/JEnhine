$Directory = "C:\Users\jay69\Data\Proj" 

$RawFolders = Get-ChildItem -Path  $Directory 

for($i = 0; $i -lt $RawFolders.Count; $i++)
{
    $FullPath = $RawFolders[$i].FullName
    $TaskFile = $FullPath+"\cbct_task.xml"
    Write-Host "$FullPath Start"
    
    # run recon
    $process = Start-Process "..\x64\Release\JEngine.exe" $TaskFile -wait -NoNewWindow -PassThru
	if($process.ExitCode -eq 0) 
    {
        Write-Host "$FullPath Pass" -ForegroundColor DarkGreen
    } 
	else
	{
        Write-Host "$FullPath FAIL" -ForegroundColor Red
        break
    }
    
    # compare results
    [xml]$xmlContents = Get-Content -Path $FullPath"\cbct_task.xml"
    $OutFolder = $xmlContents.AVR_CBCT_Reconstruction_Engine.Output_Folder.v
    $CompareFolder = $xmlContents.AVR_CBCT_Reconstruction_Engine.Compare_Folder.v

    $process = Start-Process -FilePath "..\x64\Release\Compare.exe" -ArgumentList "$OutFolder $CompareFolder" -wait -NoNewWindow -PassThru
    if($process.ExitCode -eq 0) 
    {
        Write-Host "$FullPath Pass" -ForegroundColor DarkGreen
    } 
	else
	{
        Write-Host "$FullPath FAIL" -ForegroundColor Red
        break
    }
}


