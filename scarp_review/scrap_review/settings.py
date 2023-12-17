
BOT_NAME = 'scrap_review'
SPIDER_MODULES = ['scrap_review.spiders']
NEWSPIDER_MODULE = 'scrap_review.spiders'
# Obey robots.txt rules
ROBOTSTXT_OBEY = False
SCRAPEOPS_API_KEY = '36d9ae33-4207-4d2b-9af7-1df1fbc3fcf8'
SCRAPEOPS_PROXY_ENABLED = True

DOWNLOADER_MIDDLEWARES = {
    'scrapeops_scrapy_proxy_sdk.scrapeops_scrapy_proxy_sdk.ScrapeOpsScrapyProxySdk': 725,
}
