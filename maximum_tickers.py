#!/usr/bin/env python3
"""
Максимальный список тикеров для Tiingo API
Включает все основные классы активов для максимального разнообразия данных
"""

# Основные американские индексы
US_INDICES = [
    'SPY', 'QQQ', 'IWM', 'DIA',  # Основные индексы
    'VTI', 'VTV', 'VUG', 'VEA', 'VWO',  # Широкие фонды
]

# Секторные ETF (SPDR Select Sector)
SECTOR_ETFS = [
    'XLK', 'XLF', 'XLV', 'XLI', 'XLE', 'XLU', 'XLB', 'XLP', 'XLY', 'XLRE', 'XLC'
]

# Международные рынки
INTERNATIONAL = [
    'EFA', 'EEM', 'VEA', 'VWO', 'FXI', 'EWJ', 'EWZ', 'EWG', 'EWU', 'EWC', 'EWA', 'EWH'
]

# Облигации разных типов
BONDS = [
    'TLT', 'IEF', 'SHY', 'TIP', 'LQD', 'HYG', 'JNK', 'EMB', 'AGG', 'BND'
]

# Товары и сырье
COMMODITIES = [
    'GLD', 'SLV', 'DBA', 'DBC', 'USO', 'UNG', 'PDBC', 'IAU', 'PPLT', 'PALL'
]

# Недвижимость
REAL_ESTATE = [
    'VNQ', 'VNQI', 'RWR', 'SCHH', 'IYR', 'REZ'
]

# Валютные ETF
CURRENCIES = [
    'UUP', 'FXE', 'FXY', 'FXB', 'FXC', 'FXA', 'CYB'
]

# Волатильность и защитные активы
VOLATILITY = [
    'VIX', 'VIXY', 'VXX', 'UVXY', 'SVXY'
]

# Топ крупнейшие акции
MEGA_CAPS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK.B', 'UNH', 'JNJ'
]

# Финтех и технологии
TECH_FINANCE = [
    'ARKK', 'ARKQ', 'ARKG', 'ARKW', 'FINX', 'BOTZ', 'ROBO', 'HACK', 'ICLN', 'PBW'
]

# Альтернативные стратегии
ALTERNATIVES = [
    'TAIL', 'VMOT', 'PUTW', 'CCOR', 'ALFA', 'QAI', 'DTUL', 'SWAN'
]

def get_maximum_ticker_list():
    """Возвращает максимальный список тикеров для загрузки из Tiingo"""
    all_tickers = (
        US_INDICES + SECTOR_ETFS + INTERNATIONAL + BONDS + 
        COMMODITIES + REAL_ESTATE + CURRENCIES + VOLATILITY + 
        MEGA_CAPS + TECH_FINANCE + ALTERNATIVES
    )
    # Убираем дубликаты и сортируем
    return sorted(list(set(all_tickers)))

def get_crypto_list():
    """Список криптовалют для Binance (уже есть в data_sources.py)"""
    return [
        'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'ADA-USD', 'XRP-USD', 
        'LTC-USD', 'DOGE-USD', 'AVAX-USD', 'MATIC-USD', 'DOT-USD', 'UNI-USD'
    ]

def get_priority_context_tickers():
    """Приоритетные тикеры для контекста (наиболее ликвидные и важные)"""
    return [
        # Основные индексы
        'SPY', 'QQQ', 'IWM', 'VTI',
        # Международные
        'EFA', 'EEM', 'FXI', 'EWJ',
        # Облигации  
        'TLT', 'IEF', 'LQD', 'HYG',
        # Товары
        'GLD', 'SLV', 'DBC', 'USO',
        # Валюты
        'UUP', 'FXE', 'FXY',
        # Секторы (самые важные)
        'XLK', 'XLF', 'XLV', 'XLI', 'XLE',
        # Волатильность
        'VIXY', 'VXX',
        # Топ акции
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'
    ]

if __name__ == '__main__':
    max_list = get_maximum_ticker_list()
    priority_list = get_priority_context_tickers()
    crypto_list = get_crypto_list()
    
    print(f"Максимальный список тикеров: {len(max_list)} штук")
    print(f"Приоритетный список: {len(priority_list)} штук")
    print(f"Криптовалюты: {len(crypto_list)} штук")
    print(f"Общий объем: {len(max_list) + len(crypto_list)} тикеров")
    
    print("\nПриоритетный список для запуска:")
    print(','.join(priority_list))
    
    print("\nПолный список (первые 50):")
    print(','.join(max_list[:50]))
