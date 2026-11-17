@echo off
echo Running all 16 CO1 experiments...
for %%f in (exp*.py) do (
  echo.
  echo ================= %%f =================
  python "%%f"
)
pause
