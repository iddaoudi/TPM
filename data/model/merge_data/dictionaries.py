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
    16: ["potrf", "trsm", "syrk", "gemm"]
}