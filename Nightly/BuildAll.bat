REN NuGet GTest
"%ThirdPartyLibraries%\NuGet.exe" restore ..\JEngine.sln

REM build debug
"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\msbuild.exe" ..\JEngine.sln -t:build -noLogo -property:Configuration=debug -maxcpucount -verbosity:minimal

REM build release
"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\msbuild.exe" ..\JEngine.sln -t:build -noLogo -property:Configuration=release -maxcpucount -verbosity:minimal

pause