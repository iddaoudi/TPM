metrics = ["mem_boundness", "arithm_intensity", "ilp", "l3_cache_ratio"]

cholesky_dict = {
    1: [],
    2: ["potrf"],
    3: ["gemm"],
    4: ["potrf", "gemm"],
    5: ["trsm"],
    6: ["potrf", "trsm"],
    7: ["gemm", "trsm"],
    8: ["potrf", "gemm", "trsm"],
    9: ["syrk"],
    10: ["potrf", "syrk"],
    11: ["gemm", "syrk"],
    12: ["potrf", "gemm", "syrk"],
    13: ["trsm", "syrk"],
    14: ["potrf", "trsm", "syrk"],
    15: ["gemm", "trsm", "syrk"],
    16: ["potrf", "trsm", "syrk", "gemm"],
}

qr_dict = {
    1: [],
    2: ["ormqr"],
    3: ["tsqrt"],
    4: ["ormqr", "tsqrt"],
    5: ["tsmqr"],
    6: ["ormqr", "tsmqr"],
    7: ["tsqrt", "tsmqr"],
    8: ["ormqr", "tsqrt", "tsmqr"],
    9: ["geqrt"],
    10: ["ormqr", "geqrt"],
    11: ["tsqrt", "geqrt"],
    12: ["ormqr", "tsqrt", "geqrt"],
    13: ["tsmqr", "geqrt"],
    14: ["ormqr", "tsmqr", "geqrt"],
    15: ["tsqrt", "tsmqr", "geqrt"],
    16: ["ormqr", "tsmqr", "geqrt", "tsqrt"],
}
