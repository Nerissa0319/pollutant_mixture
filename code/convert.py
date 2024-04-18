def ozone(row):
    if row.o3 <= 50:
        return row.o3 * 0.00108
    else:
        return ((row.o3 - 51) * 0.000306) + 0.055


def carbonmonoxide(row):
    if row.co <= 50:
        return row.co * 0.088
    else:
        return ((row.co - 51) * 0.1) + 4.5


def sulphurdioxide(row):
    if row.so2 <= 50:
        return row.so2 * 0.7
    else:
        return ((row.so2 - 51) * 0.7959) + 36


def nitrogendioxide(row):
    if row.no2 <= 50:
        return row.no2 * 1.06
    else:
        return ((row.no2 - 51) * 0.938) + 54


def pmtwofive(row):
    if row.pm25 <= 50:
        return row.pm25 * 0.24
    elif 51 <= row.pm25 <= 100:
        return ((row.pm25 - 51) * 0.4755) + 12.1
    elif 101 <= row.pm25 <= 150:
        return ((row.pm25 - 101) * 0.406) + 35.5
    elif 151 <= row.pm25 <= 200:
        return ((row.pm25 - 151) * 1.936) + 55.5
    else:
        return ((row.pm25 - 201) * 1.009) + 150.5


def pmten(row):
    if row.pm10 <= 50:
        return row.pm10 * 1.08
    elif 51 <= row.pm10 <= 100:
        return ((row.pm10 - 51) * 2.02) + 55
    elif 101 <= row.pm10 <= 150:
        return ((row.pm10 - 101) * 2.02) + 155
    else:
        return ((row.pm10 - 151) * 2.02) + 255