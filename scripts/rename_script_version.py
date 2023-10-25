import os

from pathlib import Path as path


for dirpath, dirnames, filenames in os.walk('./scripts'):
    renamed_file_cnt = 0
    
    for file in filenames:
        if file[-3:] != '.sh':
            continue
        
        version_name = file[:-3]
        with open(path(dirpath)/file, 'r+', encoding='utf8')as f:
            lines = list(f.readlines())
            # print(lines)
            for p, line in enumerate(lines):
                if 'version' in line:
                    if version_name not in line:
                        renamed_file_cnt += 1
                        print(f'rename script {file}')
                        lines[p] = f'    --version {version_name} \\\n'
                        f.seek(0)
                        for line in lines:
                            f.write(line)
                    break
    
    print(f'rename {renamed_file_cnt} "--version" in files')