-- ReturnFlowAI: Seller & Category Return-Risk Analytics
-- Uses window functions to compare individual seller performance against category benchmarks

-- ============================================================
-- Query 1: Seller return rate vs. category average
-- Flags sellers whose return rate significantly exceeds their category's norm
-- ============================================================

WITH seller_stats AS (
    SELECT
        seller_id,
        product_category_name,
        COUNT(*) AS total_items,
        AVG(is_return_proxy) AS seller_return_rate
    FROM order_items_full
    WHERE product_category_name IS NOT NULL
    GROUP BY seller_id, product_category_name
    HAVING COUNT(*) >= 15   -- exclude low-volume sellers to avoid noisy small-sample rates
)
SELECT
    seller_id,
    product_category_name,
    total_items,
    ROUND(seller_return_rate, 4) AS seller_return_rate,
    ROUND(AVG(seller_return_rate) OVER (PARTITION BY product_category_name), 4) AS category_avg_return_rate,
    ROUND(seller_return_rate - AVG(seller_return_rate) OVER (PARTITION BY product_category_name), 4) AS diff_from_category_avg
FROM seller_stats
ORDER BY diff_from_category_avg DESC
LIMIT 20;


-- ============================================================
-- Query 2: Rank sellers within their category by return rate
-- Identifies the single worst-performing seller(s) in each category
-- ============================================================

WITH seller_stats AS (
    SELECT
        seller_id,
        product_category_name,
        COUNT(*) AS total_items,
        AVG(is_return_proxy) AS seller_return_rate
    FROM order_items_full
    WHERE product_category_name IS NOT NULL
    GROUP BY seller_id, product_category_name
    HAVING COUNT(*) >= 15
)
SELECT
    seller_id,
    product_category_name,
    total_items,
    ROUND(seller_return_rate, 4) AS seller_return_rate,
    RANK() OVER (PARTITION BY product_category_name ORDER BY seller_return_rate DESC) AS rank_within_category
FROM seller_stats
ORDER BY product_category_name, rank_within_category;
