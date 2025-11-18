import gnss_lib_py as glp
import datetime
import urllib.request
import os

def lzw_decompress(data):
    if data[:2] != b"\x1f\x9d":
        raise ValueError("Not a valid .Z file (missing header).")
    max_bits = 12
    clear_code = 256
    end_code = 257
    code_size = 9
    dict_size = 258
    dictionary = {i: bytes([i]) for i in range(256)}
    bitpos = 16
    output = bytearray()
    def read_code():
        nonlocal bitpos
        bytepos = bitpos // 8
        bitshift = bitpos % 8
        raw = int.from_bytes(data[bytepos:bytepos+3], "big")
        raw = (raw >> (16 - bitshift - (code_size - 1))) & ((1 << code_size) - 1)
        bitpos += code_size
        return raw
    prev = b""
    while True:
        code = read_code()
        if code == clear_code:
            dictionary = {i: bytes([i]) for i in range(256)}
            dict_size = 258
            code_size = 9
            prev = b""
            continue
        if code == end_code:
            break
        if code in dictionary:
            entry = dictionary[code]
        elif code == dict_size:
            entry = prev + prev[:1]
        else:
            raise ValueError("Invalid LZW code.")
        output.extend(entry)
        if prev:
            dictionary[dict_size] = prev + entry[:1]
            dict_size += 1
            if dict_size >= (1 << code_size) and code_size < max_bits:
                code_size += 1
        prev = entry
    return bytes(output)

def get_nav_file(directory_input):
    year, month, day = map(int, directory_input.split("_")[:3])
    dt = datetime.datetime(year, month, day)
    doy = dt.timetuple().tm_yday
    yy = year % 100
    folder = f"./{directory_input}"
    os.makedirs(folder, exist_ok=True)
    nav_gz  = f"{folder}/brdc{doy:03d}0.{yy:02d}n.gz"
    nav_out = f"{folder}/brdc{doy:03d}0.{yy:02d}n"
    if os.path.exists(nav_out):
        return nav_out
    if os.path.exists(nav_gz):
        import gzip, shutil
        with gzip.open(nav_gz, "rb") as f_in, open(nav_out, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        return nav_out
    url = f"https://cddis.nasa.gov/archive/gnss/data/daily/{year}/{doy:03d}/{yy:02d}n/brdc{doy:03d}0.{yy:02d}n.gz"
    urllib.request.urlretrieve(url, nav_gz)
    import gzip, shutil
    with gzip.open(nav_gz, "rb") as f_in, open(nav_out, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return nav_out



def organize_raw_data(directory_input):
    base = f"./{directory_input}/gnss_log_{directory_input}"
    txt_file   = base + ".txt"
    nmea_file  = base + ".nmea"
    rinex_file = base + ".25o"
    nav_file   = get_nav_file(directory_input)
    txt_raw   = glp.AndroidRawGnss(txt_file)
    nmea_raw  = glp.Nmea(nmea_file)
    rinex_raw = glp.RinexObs(rinex_file)
    nav_raw   = glp.RinexNav(nav_file)
    txt_data = txt_raw.preprocess(txt_file)
    return txt_data, nmea_raw.data, rinex_raw, nav_raw
