def get_json_list(path):
    import json
    f = open(path, 'r')
    info = []
    for line in f.readlines():
        info.append(json.loads(line))
    return info

def write_line_to_file(in_str, f):
    f.write(in_str)
    f.write("\n")
    f.flush()

def split_string(string, length):
    return [string[i:i+length] for i in range(0, len(string), length)]

def auto_new_line(in_str):
    return "\n".join(split_string(in_str, 100))