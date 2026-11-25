import subprocess, re

def get_quota_meter():
    res = subprocess.run([r'C:\Python312\Scripts\kaggle.exe', 'b', 'quota'], capture_output=True, text=True)
    out = res.stdout
    m = re.search(r'Daily\s+\$([\d\.]+)\s+\$([\d\.]+)\s+\$([\d\.]+)', out)
    if m:
        used = float(m.group(1))
        remaining = float(m.group(2))
        total = float(m.group(3))
        return {'used': used, 'remaining': remaining, 'total': total}
    return None

if __name__ == '__main__':
    print('Quota meter:', get_quota_meter())
