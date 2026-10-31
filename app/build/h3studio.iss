; Inno Setup script: ISCC.exe app\build\h3studio.iss  (after the PyInstaller build)
#define AppName "Minimax H3 Workspace"
#define AppVersion "1.0.0"
#define AppExe "MinimaxH3Workspace.exe"

[Setup]
AppId={{6B0E3C1E-5D7A-4C64-9A0B-7E2B1C9F4A31}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppName}
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; per-user install: no admin rights needed, and the app can write next to itself
PrivilegesRequired=lowest
OutputDir=output
OutputBaseFilename=Minimax.H3.Workspace-Setup
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern

[Languages]
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "dist\MinimaxH3Workspace\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

; ComfyUI, models and settings live outside {app} (chosen in the app / %LOCALAPPDATA%\MinimaxH3Workspace)
; and are kept on uninstall on purpose: they are tens of gigabytes.
