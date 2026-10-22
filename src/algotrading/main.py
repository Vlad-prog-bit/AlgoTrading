"""AlgoTrading: каркас исследовательского проекта."""
import asyncio
from parser.report_parser import parse_companies
from parser.video_parser import video_parser
import platform
from preprocessing.text_reader import process_reports
system = platform.system()

COMPANIES = {
    "sber": 3043,
    "tbank": 2989,  # АО ТБанк, именно банк, не холдинговая компания.
    "vk": 38965,   # МКПАО ВК; старый архив VK Company Limited — 38185.
    "gazprom": 934,
    "megafon": 219,
}
async def start_parser():
    '''
    здесь начинается вызов
    функций парсинга
    '''
    report_types=(2,3,4)
    years=[2024]
    companies_to_pars = ["sber","tbank"]
    return await parse_companies(
        companies={name: COMPANIES[name] for name in companies_to_pars},
        years=years,  # None — все годы
        report_types=report_types,  # 2 — годовые, 3 — бухгалтерские, 4 — консолидированные
    )
if __name__ == '__main__':
    # asyncio.run(start_parser())
    # video_parser(count=2,topic='финансовая отчетность МСФО анали')
    #if system == "Darwin":
    #    from preprocessing.transcript_mac import transcript_mac
    #    transcript_mac()
    #else:
    #    from preprocessing.transcript_windows import transcript_windows
    #    transcript_windows()
    process_reports(
        companies=["sber", "tbank"],
        report_types=[
            "annual_reports",
            "financial_statements",
            "consolidated_reports",
    ],
    )