"""スキャン対象の銘柄リスト(プリセット)。"""

UNIVERSES = {
    "jp_major": {
        "name": "日本株 主要銘柄",
        "tickers": {
            "7203.T": "トヨタ自動車", "6758.T": "ソニーグループ", "9984.T": "ソフトバンクグループ",
            "8306.T": "三菱UFJ FG", "6861.T": "キーエンス", "9432.T": "NTT",
            "6098.T": "リクルートHD", "8035.T": "東京エレクトロン", "9983.T": "ファーストリテイリング",
            "4063.T": "信越化学工業", "6501.T": "日立製作所", "7974.T": "任天堂",
            "8316.T": "三井住友FG", "4502.T": "武田薬品工業", "6367.T": "ダイキン工業",
            "7267.T": "ホンダ", "8058.T": "三菱商事", "8031.T": "三井物産",
            "8001.T": "伊藤忠商事", "9433.T": "KDDI", "6902.T": "デンソー",
            "6981.T": "村田製作所", "4568.T": "第一三共", "6954.T": "ファナック",
            "7741.T": "HOYA", "6594.T": "ニデック", "4519.T": "中外製薬",
            "8766.T": "東京海上HD", "8411.T": "みずほFG", "6273.T": "SMC",
            "6146.T": "ディスコ", "6857.T": "アドバンテスト", "7011.T": "三菱重工業",
            "6702.T": "富士通", "6752.T": "パナソニックHD", "4661.T": "オリエンタルランド",
            "2914.T": "日本たばこ産業", "9022.T": "JR東海", "5108.T": "ブリヂストン",
            "7751.T": "キヤノン",
        },
    },
    "us_major": {
        "name": "米国株 主要銘柄",
        "tickers": {
            "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon",
            "GOOGL": "Alphabet", "META": "Meta", "TSLA": "Tesla", "AVGO": "Broadcom",
            "BRK-B": "Berkshire Hathaway", "JPM": "JPMorgan Chase", "V": "Visa",
            "UNH": "UnitedHealth", "XOM": "Exxon Mobil", "MA": "Mastercard",
            "JNJ": "Johnson & Johnson", "PG": "Procter & Gamble", "HD": "Home Depot",
            "COST": "Costco", "LLY": "Eli Lilly", "ABBV": "AbbVie", "MRK": "Merck",
            "KO": "Coca-Cola", "PEP": "PepsiCo", "ADBE": "Adobe", "CRM": "Salesforce",
            "NFLX": "Netflix", "AMD": "AMD", "INTC": "Intel", "ORCL": "Oracle",
            "WMT": "Walmart",
        },
    },
}


def ticker_name(ticker: str) -> str:
    for u in UNIVERSES.values():
        if ticker in u["tickers"]:
            return u["tickers"][ticker]
    return ""
