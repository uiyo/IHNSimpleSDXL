import os

win32_root = os.path.dirname(os.path.dirname(__file__))
python_embeded_path = os.path.join(win32_root, 'python_embeded')

is_win32_standalone_build = os.path.exists(python_embeded_path) and os.path.isdir(python_embeded_path)

win32_cmd = '''
@echo off
.\python_embeded\python.exe -s SimpleSDXL\{cmds} %*
echo All done.
pause
'''


def build_launcher():
    if not is_win32_standalone_build:
        return

    branchs = {"SimpleSDXL": "entry_with_update.py", "SimpleSDXL_dev": "entry_with_update.py --dev", "SimpleSDXL_without_update": "launch.py", "SimpleSDXL_commit": "launch_with_commit.py 56e5200"}

    for (name, cmd) in branchs.items():
        win32_cmd_preset = win32_cmd.replace('{cmds}', f'{cmd}')
        bat_path = os.path.join(win32_root, f'run_{name}.bat')
        if not os.path.exists(bat_path) or bat_path=='run_SimpleSDXL_commit.bat':
            with open(bat_path, "w", encoding="utf-8") as f:
                f.write(win32_cmd_preset)
    return
