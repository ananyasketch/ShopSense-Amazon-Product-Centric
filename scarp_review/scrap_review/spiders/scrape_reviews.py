import scrapy
from urllib.parse import urlparse, unquote

class AmazonReviewsSpider(scrapy.Spider):
    name = "amazon_reviews"

    def start_requests(self):
        url = getattr(self, 'url', None)  # Get URL from spider argument
        asin = getattr(self, 'asin', None)  # Get ASIN from spider argument
        yield scrapy.Request(url=url, callback=self.parse_reviews,
                             meta={'asin': asin, 'retry_count': 0})

    def parse_reviews(self, response):
        asin = response.meta['asin']
        retry_count = response.meta['retry_count']

        # Next Page Url
        next_page_relative_url = response.css(".a-pagination .a-last > a::attr(href)").get()
        if next_page_relative_url is not None:
            retry_count = 0
            next_page = response.urljoin(next_page_relative_url)
            yield scrapy.Request(url=next_page, callback=self.parse_reviews,
                                 meta={'asin': asin, 'retry_count': retry_count})
        elif retry_count < 3:
            retry_count = retry_count + 1
            yield scrapy.Request(url=response.url, callback=self.parse_reviews, dont_filter=False,
                                 meta={'asin': asin, 'retry_count': retry_count})

        # Product Reviews
        review_elements = response.css("#cm_cr-review_list div.review")
        for review_element in review_elements:
            yield {
                "text": "".join(review_element.css("span[data-hook=review-body] ::text").getall()).strip(),
                "title": review_element.css("*[data-hook=review-title]>span::text").get(),
                "location_and_date": review_element.css("span[data-hook=review-date] ::text").get(),
                "verified": bool(review_element.css("span[data-hook=avp-badge] ::text").get()),
                "rating": review_element.css("*[data-hook*=review-star-rating] ::text").re(r"(\d+\.*\d*) out")[0],
            }
