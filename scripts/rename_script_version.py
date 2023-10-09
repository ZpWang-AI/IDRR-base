import os

from pathlib import Path as path


version_s = 'version'

for dirpath, dirnames, filenames in os.walk('./scripts'):
    for file in filenames:
        if file[-3:] == '.sh':
            version_name = file[:-3]
            with open(path(dirpath)/file, 'r+', encoding='utf8')as f:
                lines = list(f.readlines())
                # print(lines)
                for p, line in enumerate(lines):
                    if version_s in line:
                        if version_name not in line:
                            print(f'rename script {file}')
                            f.seek(0)
                            for line in lines:
                                f.write(line)
                            lines[p] = f'    --version {version_name} \\\n'
                        break
                