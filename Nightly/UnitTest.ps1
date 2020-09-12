$Directory = "..\x64\" 

if( -not( Test-Path -Path $Directory))
{
	$Directory = ".\"
}

$ExeFiles = Get-ChildItem -Path  $Directory -Recurse -Include UT*.exe


$FailCount = 0
for($i = 0; $i -lt $ExeFiles.Count; $i++)
{
	$FullName = $ExeFiles[$i].FullName
	$BaseName = $ExeFiles[$i].BaseName
	
	Write-Host $FullName
	
	$process = Start-Process $ExeFiles[$i].FullName -wait -NoNewWindow -PassThru
	
	if($process.ExitCode -eq 0) 
    {
		Write-Host "$BaseName Pass" -ForegroundColor DarkGreen
    } 
	else
	{
		Write-Host "$BaseName FAIL" -ForegroundColor Red
		$FailCount = $FailCount + 1
	}
}


if ($FailCount -ne 0)
{
	EXIT(1)
}
else
{
	EXIT(0)
}