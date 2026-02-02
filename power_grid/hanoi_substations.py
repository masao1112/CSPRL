"""
Dữ liệu trạm biến áp thực tế tại Hà Nội

Nguồn: Google Maps - Electrical Substations in Hanoi
Tọa độ được chuyển đổi từ Plus Code sang lat/lon.
"""

# Danh sách trạm biến áp 500kV
SUBSTATIONS_500KV = [
    {
        "name": "TBA 500kV Tây Hà Nội",
        "plus_code": "XJMW+R23",
        "lat": 20.9745,
        "lon": 105.6842,
        "capacity_mva": 900,
        "status": "active",
    },
    {
        "name": "TBA 500kV Đông Anh",
        "plus_code": "5V7X+62J",
        "lat": 21.1678,
        "lon": 105.8523,
        "capacity_mva": 900,
        "status": "active",
    },
    {
        "name": "TBA 500kV Thường Tín",
        "plus_code": "VVH3+8XJ",
        "lat": 20.8312,
        "lon": 105.8634,
        "capacity_mva": 900,
        "status": "active",
    },
]

# Danh sách trạm biến áp 220kV
SUBSTATIONS_220KV = [
    {
        "name": "TBA 220kV Hà Đông",
        "plus_code": "XQ55+H2H",
        "lat": 20.9712,
        "lon": 105.7523,
        "address": "Ng. 14, Hà Đông",
        "capacity_mva": 500,
        "status": "active",
    },
    {
        "name": "TBA 220kV Mai Động",
        "plus_code": None,
        "lat": 20.9923,
        "lon": 105.8712,
        "address": "Mai Động, Hoàng Mai",
        "capacity_mva": 500,
        "status": "active",
    },
    {
        "name": "TBA 220kV Thành Công",
        "plus_code": "2R8C+572",
        "lat": 21.0245,
        "lon": 105.8134,
        "address": "P. Đặng Tiến Đông",
        "capacity_mva": 500,
        "status": "active",
    },
    {
        "name": "TBA 220kV Chèm",
        "plus_code": None,
        "lat": 21.1012,
        "lon": 105.8023,
        "address": "59 Đ. Yên Nội",
        "capacity_mva": 500,
        "status": "active",
    },
    {
        "name": "TBA 220kV Tây Hồ",
        "plus_code": None,
        "lat": 21.0678,
        "lon": 105.8234,
        "address": "Ngõ 15 An Dương Vương",
        "capacity_mva": 500,
        "status": "active",
    },
    {
        "name": "TBA 220kV Long Biên",
        "plus_code": "3W88+PQJ",
        "lat": 21.0512,
        "lon": 105.9123,
        "address": "Đường Đê Vàng",
        "capacity_mva": 500,
        "status": "active",
    },
    {
        "name": "TBA 220kV Xuân Mai",
        "plus_code": "WHP8+J74",
        "lat": 20.8534,
        "lon": 105.5712,
        "capacity_mva": 500,
        "status": "active",
    },
    {
        "name": "TBA 220kV Vân Trì",
        "plus_code": "5QFF+R2X",
        "lat": 21.1234,
        "lon": 105.7823,
        "address": "Đ. Mê Linh",
        "capacity_mva": 500,
        "status": "active",
    },
    {
        "name": "TBA 220kV Sơn Tây",
        "plus_code": "4FJ8+HMP",
        "lat": 21.1367,
        "lon": 105.5045,
        "capacity_mva": 500,
        "status": "active",
    },
]

# Danh sách trạm biến áp 110kV (chi tiết hơn, phục vụ cho mô phỏng)
# District codes: hoan_kiem, dong_da, ba_dinh, hai_ba_trung, thanh_xuan,
#                 cau_giay, tay_ho, long_bien, hoang_mai, ha_dong,
#                 nam_tu_liem, bac_tu_liem, thanh_tri, gia_lam, dong_anh, soc_son
SUBSTATIONS_110KV = [
    # Khu vực Hoàn Kiếm
    {
        "name": "TBA 110kV Trần Hưng Đạo (E1.12)",
        "plus_code": "2V95+GXC",
        "lat": 21.0223,
        "lon": 105.8512,
        "address": "Trần Hưng Đạo, Phan Chu Trinh",
        "district": "hoan_kiem",
        "capacity_mva": 63,
        "status": "active",
    },
    {
        "name": "TBA 110kV E1.18 Bờ Hồ",
        "lat": 21.0280,
        "lon": 105.8560,
        "address": "781 Hồng Hà, Chương Dương Độ",
        "district": "hoan_kiem",
        "capacity_mva": 63,
        "status": "active",
    },
    
    # Khu vực Đống Đa
    {
        "name": "TBA 110kV Văn Chương 1",
        "lat": 21.0156,
        "lon": 105.8345,
        "address": "114 Ng. Trung Tả, Khâm Thiên",
        "district": "dong_da",
        "capacity_mva": 40,
        "status": "active",
    },
    {
        "name": "TBA 110kV Trung Tự 2",
        "lat": 21.0089,
        "lon": 105.8378,
        "address": "24 Ng. 46 P. Phạm Ngọc Thạch, Trung Tự",
        "district": "dong_da",
        "capacity_mva": 40,
        "status": "active",
    },
    {
        "name": "TBA 110kV Công viên Thống Nhất",
        "lat": 21.0178,
        "lon": 105.8423,
        "address": "Gần công viên Thống Nhất",
        "district": "dong_da",
        "capacity_mva": 63,
        "status": "active",
    },
    
    # Khu vực Ba Đình
    {
        "name": "TBA 110kV Yên Phụ",
        "lat": 21.0456,
        "lon": 105.8423,
        "address": "50 P. Châu Long, Trúc Bạch",
        "district": "ba_dinh",
        "capacity_mva": 63,
        "status": "active",
    },
    {
        "name": "TBA 110kV Bắc Thành Công (E1.63)",
        "lat": 21.0312,
        "lon": 105.8156,
        "address": "Phường Thành Công",
        "district": "ba_dinh",
        "capacity_mva": 63,
        "status": "active",
    },
    
    # Khu vực Hai Bà Trưng
    {
        "name": "TBA 110kV Thanh Nhàn",
        "lat": 21.0034,
        "lon": 105.8589,
        "address": "Ngõ 84 P. Kim Ngưu, Thanh Nhàn",
        "district": "hai_ba_trung",
        "capacity_mva": 63,
        "status": "active",
    },
    
    # Khu vực Thanh Xuân
    {
        "name": "TBA 110kV Thượng Đình",
        "lat": 21.0034,
        "lon": 105.8089,
        "address": "23 Ng. 282 Đ. Khương Đình, Hạ Đình",
        "district": "thanh_xuan",
        "capacity_mva": 63,
        "status": "active",
    },
    
    # Khu vực Nam Từ Liêm
    {
        "name": "TBA 110kV Thanh Xuân E1.20",
        "plus_code": "2Q2R+8PJ",
        "lat": 21.0112,
        "lon": 105.8145,
        "address": "Trung Văn",
        "district": "nam_tu_liem",
        "capacity_mva": 63,
        "status": "active",
    },
    {
        "name": "TBA 110kV Cầu Diễn",
        "lat": 21.0389,
        "lon": 105.7634,
        "address": "Lô A2 CN8, Phúc Diễn",
        "district": "nam_tu_liem",
        "capacity_mva": 63,
        "status": "active",
    },
    {
        "name": "TBA 110kV Mỹ Đình (E1.25)",
        "lat": 21.0134,
        "lon": 105.7645,
        "address": "Đường Đình Thôn, Mễ Trì",
        "district": "nam_tu_liem",
        "capacity_mva": 63,
        "status": "active",
    },
    
    # Khu vực Hà Đông
    {
        "name": "TBA 110kV Dương Nội (E1.61)",
        "plus_code": "XQJ5+5X5",
        "lat": 20.9756,
        "lon": 105.7234,
        "district": "ha_dong",
        "capacity_mva": 63,
        "status": "active",
    },
    {
        "name": "TBA 110kV Mỗ Lao",
        "lat": 20.9823,
        "lon": 105.7567,
        "district": "ha_dong",
        "capacity_mva": 63,
        "status": "active",
    },
    {
        "name": "TBA 110kV Đồng Mai",
        "plus_code": "WQJ2+4F6",
        "lat": 20.9023,
        "lon": 105.7234,
        "district": "ha_dong",
        "capacity_mva": 40,
        "status": "active",
    },
    
    # Khu vực Cầu Giấy
    {
        "name": "TBA 110kV Nghĩa Đô",
        "plus_code": "2QRW+VMV",
        "lat": 21.0412,
        "lon": 105.7978,
        "district": "cau_giay",
        "capacity_mva": 63,
        "status": "active",
    },
    
    # Khu vực Bắc Từ Liêm (moved from cau_giay)
    {
        "name": "TBA 110kV Từ Liêm",
        "plus_code": "3Q6C+G83",
        "lat": 21.0534,
        "lon": 105.7523,
        "address": "Cổ Nhuế 2",
        "district": "bac_tu_liem",
        "capacity_mva": 63,
        "status": "active",
    },
    
    # Khu vực Hoài Đức
    {
        "name": "TBA 110kV Trôi (E1.31)",
        "plus_code": "3P86+JVR",
        "lat": 21.0623,
        "lon": 105.7134,
        "district": "hoai_duc",
        "capacity_mva": 40,
        "status": "active",
    },
    
    # Khu vực Long Biên
    {
        "name": "TBA 110kV Sài Đồng",
        "plus_code": "2WH3+M2R",
        "lat": 21.0278,
        "lon": 105.9012,
        "district": "long_bien",
        "capacity_mva": 63,
        "status": "active",
    },
    {
        "name": "TBA 110kV NC Long Biên (E1.47)",
        "plus_code": "3W88+HGM",
        "lat": 21.0512,
        "lon": 105.9089,
        "district": "long_bien",
        "capacity_mva": 63,
        "status": "active",
    },
    {
        "name": "TBA 110kV Ngọc Thụy",
        "lat": 21.0567,
        "lon": 105.8678,
        "address": "Phường Ngọc Thụy",
        "district": "long_bien",
        "capacity_mva": 63,
        "status": "active",
    },
    
    # Khu vực Gia Lâm
    {
        "name": "TBA 110kV Gia Lâm 2",
        "lat": 21.0345,
        "lon": 105.9234,
        "address": "Thôn Phú Thụy, xã Phú Thị",
        "district": "gia_lam",
        "capacity_mva": 40,
        "status": "active",
    },
    {
        "name": "TBA 110kV Đông Dư",
        "plus_code": "XWXC+F87",
        "lat": 20.9812,
        "lon": 105.9312,
        "district": "gia_lam",
        "capacity_mva": 40,
        "status": "active",
    },
    
    # Khu vực Thanh Trì
    {
        "name": "TBA 110kV Đại Kim",
        "plus_code": "XR93+GQW",
        "lat": 20.9878,
        "lon": 105.8356,
        "address": "Thanh Liệt",
        "district": "thanh_tri",
        "capacity_mva": 63,
        "status": "active",
    },
    
    # Khu vực Hoàng Mai
    {
        "name": "TBA 110kV Linh Đàm",
        "lat": 20.9612,
        "lon": 105.8234,
        "address": "56 P. Linh Đường, Hoàng Liệt",
        "district": "hoang_mai",
        "capacity_mva": 63,
        "status": "active",
    },
    {
        "name": "TBA 110kV Lĩnh Nam",
        "plus_code": "XQJ6+522",
        "lat": 20.9734,
        "lon": 105.8745,
        "address": "467 Đ. Lĩnh Nam",
        "district": "hoang_mai",
        "capacity_mva": 40,
        "status": "active",
    },
    {
        "name": "TBA 110kV Hồ Yên Sở (E1.64)",
        "plus_code": "XV52+76J",
        "lat": 20.9645,
        "lon": 105.8567,
        "district": "hoang_mai",
        "capacity_mva": 40,
        "status": "active",
    },
    {
        "name": "TBA 110kV Mai Động (E1.3)",
        "lat": 20.9923,
        "lon": 105.8712,
        "address": "Khu vực 220kV Mai Động",
        "district": "hoang_mai",
        "capacity_mva": 63,
        "status": "active",
    },
    {
        "name": "TBA 110kV Minh Khai",
        "lat": 20.9834,
        "lon": 105.8656,
        "address": "Phường Vĩnh Hưng",
        "district": "hoang_mai",
        "capacity_mva": 63,
        "status": "active",
    },
    {
        "name": "TBA 110kV Phương Liệt GIS",
        "lat": 20.9912,
        "lon": 105.8523,
        "address": "Phường Phương Liệt",
        "district": "hoang_mai",
        "capacity_mva": 63,
        "status": "active",
    },
    
    # Khu vực Tây Hồ
    {
        "name": "TBA 110kV Nam Thăng Long",
        "lat": 21.0678,
        "lon": 105.8245,
        "address": "20 An Dương Vương",
        "district": "tay_ho",
        "capacity_mva": 63,
        "status": "active",
    },
    {
        "name": "TBA 110kV Nhật Tân",
        "lat": 21.0789,
        "lon": 105.8312,
        "address": "Ngõ 15 An Dương Vương",
        "district": "tay_ho",
        "capacity_mva": 40,
        "status": "active",
    },
    
    # Khu vực Đông Anh
    {
        "name": "TBA 110kV Mai Lâm",
        "plus_code": "4R4X+3G8",
        "lat": 21.0923,
        "lon": 105.8534,
        "address": "Xã Xuân Canh",
        "district": "dong_anh",
        "capacity_mva": 63,
        "status": "active",
    },
    {
        "name": "TBA 110kV Hải Bối",
        "plus_code": "4Q6X+J7V",
        "lat": 21.0812,
        "lon": 105.7989,
        "district": "dong_anh",
        "capacity_mva": 40,
        "status": "active",
    },
    {
        "name": "TBA 110kV Kim Chung (E1.72)",
        "plus_code": "3P57+78",
        "lat": 21.0678,
        "lon": 105.7234,
        "district": "dong_anh",
        "capacity_mva": 40,
        "status": "active",
    },
    {
        "name": "TBA 110kV E1.1 Đông Anh",
        "lat": 21.1234,
        "lon": 105.8312,
        "district": "dong_anh",
        "capacity_mva": 63,
        "status": "active",
    },
    {
        "name": "TBA 110kV Bắc Thăng Long",
        "address": "KCN Thăng Long, Đông Anh",
        "lat": 21.1090,
        "lon": 105.7830,
        "district": "dong_anh",
        "capacity_mva": 63,
        "status": "active",
    },
    {
        "name": "TBA 220kV Vân Trì (Source 110kV)",
        "address": "Xã Nam Hồng, Đông Anh",
        "lat": 21.1500,
        "lon": 105.8160,
        "district": "dong_anh",
        "capacity_mva": 126,
        "status": "active",
    },
    
    # Khu vực Sóc Sơn
    {
        "name": "TBA 110kV Nội Bài",
        "plus_code": "6RH7+P5J",
        "lat": 21.2234,
        "lon": 105.8023,
        "district": "soc_son",
        "capacity_mva": 63,
        "status": "active",
    },
    {
        "name": "TBA 110kV Sóc Sơn 2",
        "plus_code": "6VVM+M5",
        "lat": 21.2456,
        "lon": 105.8623,
        "address": "Thôn Đại Tảo, xã Xuân Giang",
        "district": "soc_son",
        "capacity_mva": 63,
        "status": "active",
    },
    {
        "name": "TBA 110kV KCN Nội Bài",
        "address": "Xã Quang Tiến, Sóc Sơn",
        "lat": 21.2000,
        "lon": 105.8330,
        "district": "soc_son",
        "capacity_mva": 40,
        "status": "active",
    },
    
    # Khu vực ngoại thành khác
    {
        "name": "TBA 110kV Thường Tín",
        "plus_code": "VVJF+WH6",
        "lat": 20.8456,
        "lon": 105.8678,
        "district": "hanoi_other",
        "capacity_mva": 40,
        "status": "active",
    },
    {
        "name": "TBA 110kV Thanh Oai",
        "plus_code": "VQ79+628",
        "lat": 20.8678,
        "lon": 105.7834,
        "district": "hanoi_other",
        "capacity_mva": 40,
        "status": "active",
    },
    {
        "name": "TBA 110kV Phúc Thọ",
        "plus_code": "3JP6+89F",
        "lat": 21.0978,
        "lon": 105.5612,
        "district": "hanoi_other",
        "capacity_mva": 40,
        "status": "active",
    },
    {
        "name": "TBA 110kV Vân Đình",
        "plus_code": "PQJF+GXC",
        "lat": 20.7623,
        "lon": 105.7834,
        "district": "hanoi_other",
        "capacity_mva": 40,
        "status": "active",
    },
    {
        "name": "TBA 110kV Bắc An Khánh",
        "plus_code": "2PFG+224",
        "lat": 21.0089,
        "lon": 105.7234,
        "district": "hoai_duc",
        "capacity_mva": 40,
        "status": "active",
    },
    {
        "name": "TBA 110kV Thạch Thất",
        "plus_code": "3HV4+RW8",
        "lat": 21.0723,
        "lon": 105.5545,
        "district": "hanoi_other",
        "capacity_mva": 40,
        "status": "active",
    },
    {
        "name": "TBA 110kV Thạch Thất 2 (E1.74)",
        "plus_code": "3HGC+894",
        "lat": 21.0645,
        "lon": 105.5423,
        "district": "hanoi_other",
        "capacity_mva": 40,
        "status": "active",
    },
    {
        "name": "TBA 110kV Phùng Xá",
        "plus_code": "2J4F+X5H",
        "lat": 21.0234,
        "lon": 105.6512,
        "district": "hanoi_other",
        "capacity_mva": 40,
        "status": "active",
    },
    {
        "name": "TBA 100kV Quốc Oai",
        "lat": 20.9856,
        "lon": 105.6234,
        "address": "Xã Đồng Quang, Quốc Oai",
        "district": "hanoi_other",
        "capacity_mva": 63,
        "status": "active",
    },
    {
        "name": "TBA 110kV CNC2 (E1.73)",
        "plus_code": "XHP3+MQ8",
        "lat": 20.9556,
        "lon": 105.5789,
        "district": "hanoi_other",
        "capacity_mva": 40,
        "status": "active",
    },
    {
        "name": "TBA 110kV E1.7 Sơn Tây",
        "lat": 21.1330,
        "lon": 105.5000,
        "address": "Thị xã Sơn Tây",
        "district": "hanoi_other",
        "capacity_mva": 40,
        "status": "active",
    },
    {
        "name": "TBA 110kV KCN Phú Nghĩa",
        "address": "KCN Phú Nghĩa, Chương Mỹ",
        "lat": 20.9167,
        "lon": 105.5833,
        "district": "hanoi_other",
        "capacity_mva": 40,
        "status": "active",
    },
    {
        "name": "TBA 110kV Quang Minh",
        "address": "KCN Quang Minh, Mê Linh",
        "lat": 21.1833,
        "lon": 105.7833,
        "district": "hanoi_other",
        "capacity_mva": 63,
        "status": "active",
    },
]


def get_substations_by_district(district: str, voltage_levels: list = None):
    """
    Lấy danh sách trạm biến áp theo quận.
    
    Args:
        district: Tên quận (cau_giay, dong_da, hoan_kiem, thanh_xuan, ha_dong, long_bien, tay_ho)
        voltage_levels: Danh sách cấp điện áp cần lấy (110, 220, 500). None = tất cả.
        
    Returns:
        List các trạm biến áp
    """
    result = []
    
    if voltage_levels is None or 500 in voltage_levels:
        result.extend(SUBSTATIONS_500KV)
    
    if voltage_levels is None or 220 in voltage_levels:
        result.extend(SUBSTATIONS_220KV)
    
    if voltage_levels is None or 110 in voltage_levels:
        for sub in SUBSTATIONS_110KV:
            if district is None or sub.get("district") == district:
                result.append(sub)
    
    return result


def get_all_substations():
    """Lấy tất cả trạm biến áp."""
    return SUBSTATIONS_500KV + SUBSTATIONS_220KV + SUBSTATIONS_110KV
