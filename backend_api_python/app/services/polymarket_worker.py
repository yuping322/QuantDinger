"""
Polymarket后台任务
每30分钟更新一次市场数据，并批量分析市场机会
"""
import os
import json
import argparse
import threading
import time
from datetime import datetime
from typing import List, Dict, Optional, Any

# 支持直接执行该文件：自动补齐 backend_api_python 到 sys.path
if __package__ in (None, ""):
    import sys
    _this_file_dir = os.path.dirname(os.path.abspath(__file__))
    _backend_root_dir = os.path.dirname(os.path.dirname(_this_file_dir))
    if _backend_root_dir not in sys.path:
        sys.path.insert(0, _backend_root_dir)

from app.utils.logger import get_logger
from app.data_sources.polymarket import PolymarketDataSource
from app.services.polymarket_batch_analyzer import PolymarketBatchAnalyzer

logger = get_logger(__name__)


class PolymarketWorker:
    """Polymarket数据更新和分析后台任务"""
    
    def __init__(self, update_interval_minutes: int = 30, analysis_cache_minutes: int = 1440):  # 24小时缓存
        """
        初始化后台任务
        
        Args:
            update_interval_minutes: 市场数据更新间隔（分钟）
            analysis_cache_minutes: AI分析结果缓存时间（分钟）
        """
        self.update_interval_minutes = update_interval_minutes
        self.analysis_cache_minutes = analysis_cache_minutes
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.polymarket_source = PolymarketDataSource()
        self.batch_analyzer = PolymarketBatchAnalyzer()
        self._last_update_ts = 0.0
        
    def start(self) -> bool:
        """启动后台任务"""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return True
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, name="PolymarketWorker", daemon=True)
            self._thread.start()
            logger.info(f"PolymarketWorker started (update_interval={self.update_interval_minutes}min, cache={self.analysis_cache_minutes}min)")
            return True
    
    def stop(self, timeout_sec: float = 5.0) -> None:
        """停止后台任务"""
        with self._lock:
            if not self._thread or not self._thread.is_alive():
                return
            self._stop_event.set()
            self._thread.join(timeout=timeout_sec)
            if self._thread.is_alive():
                logger.warning("PolymarketWorker thread did not stop within timeout")
            else:
                logger.info("PolymarketWorker stopped")
    
    def _run_loop(self) -> None:
        """主循环"""
        logger.info("PolymarketWorker loop started")
        
        # 启动时立即执行一次
        self._update_markets_and_analyze()
        
        while not self._stop_event.is_set():
            try:
                # 等待指定时间间隔
                wait_seconds = self.update_interval_minutes * 60
                if self._stop_event.wait(wait_seconds):
                    break  # 如果收到停止信号，退出循环
                
                # 执行更新和分析
                self._update_markets_and_analyze()
                
            except Exception as e:
                logger.error(f"PolymarketWorker loop error: {e}", exc_info=True)
                # 出错后等待1分钟再重试
                self._stop_event.wait(60)
        
        logger.info("PolymarketWorker loop stopped")
    
    def _fetch_unique_markets(self, categories: Optional[List[str]] = None, per_category_limit: int = 50) -> List[Dict[str, Any]]:
        """从多个分类抓取并按 market_id 去重。"""
        if categories is None:
            categories = ["crypto", "politics", "economics", "sports", "tech", "finance", "geopolitics", "culture", "climate", "entertainment"]

        all_markets: List[Dict[str, Any]] = []
        for category in categories:
            try:
                markets = self.polymarket_source.get_trending_markets(category, limit=per_category_limit)
                all_markets.extend(markets)
                logger.info(f"Fetched {len(markets)} markets from category: {category}")
            except Exception as e:
                logger.warning(f"Failed to fetch markets for category {category}: {e}")

        unique_markets: Dict[str, Dict[str, Any]] = {}
        for market in all_markets:
            market_id = market.get('market_id')
            if market_id:
                unique_markets[market_id] = market

        logger.info(f"Total unique markets: {len(unique_markets)}")
        return list(unique_markets.values())

    def _select_rule_based_opportunities(self, markets: List[Dict[str, Any]], max_opportunities: int = 30) -> List[Dict[str, Any]]:
        """规则筛选高价值机会，减少LLM调用。"""
        rule_based_opportunities = []
        for market in markets:
            prob = market.get('current_probability', 50.0)
            volume = market.get('volume_24h', 0)
            divergence = abs(prob - 50.0)
            if volume > 5000 and divergence > 8:
                rule_based_opportunities.append(market)

        if not rule_based_opportunities:
            return []

        rule_based_opportunities.sort(
            key=lambda x: (x.get('volume_24h', 0) * abs(x.get('current_probability', 50) - 50)),
            reverse=True
        )
        return rule_based_opportunities[:max_opportunities]

    def run_once_analysis(self, categories: Optional[List[str]] = None, per_category_limit: int = 50,
                          max_opportunities: int = 30, save_to_db: bool = True) -> Dict[str, Any]:
        """运行一次完整流程：抓取市场 -> 规则筛选 -> LLM分析 -> 可选入库。"""
        start_time = time.time()

        markets = self._fetch_unique_markets(categories=categories, per_category_limit=per_category_limit)
        logger.info(f"Starting batch analysis for {len(markets)} markets...")

        opportunities = self._select_rule_based_opportunities(markets, max_opportunities=max_opportunities)
        if opportunities:
            logger.info(f"Rule-based filtering: {len(opportunities)} opportunities, analyzing top {len(opportunities)} with LLM")
            analyzed_markets = self.batch_analyzer.batch_analyze_markets(
                opportunities,
                max_opportunities=max_opportunities,
            )
        else:
            logger.info("No rule-based opportunities found, skipping LLM analysis")
            analyzed_markets = []

        if save_to_db and analyzed_markets:
            self.batch_analyzer.save_batch_analysis(analyzed_markets)

        elapsed = time.time() - start_time
        summary = {
            "markets_fetched": len(markets),
            "opportunities_selected": len(opportunities),
            "opportunities_analyzed": len(analyzed_markets),
            "saved_to_db": bool(save_to_db and analyzed_markets),
            "elapsed_seconds": round(elapsed, 2),
            "ran_at": datetime.utcnow().isoformat() + "Z",
        }
        logger.info(
            f"Polymarket run-once completed: fetched={summary['markets_fetched']}, "
            f"selected={summary['opportunities_selected']}, analyzed={summary['opportunities_analyzed']}, "
            f"elapsed={summary['elapsed_seconds']}s"
        )
        self._last_update_ts = time.time()
        return summary

    def _update_markets_and_analyze(self) -> None:
        """更新市场数据并分析（后台循环复用 run_once 核心逻辑）。"""
        try:
            logger.info("Starting Polymarket data update and analysis...")
            self.run_once_analysis(save_to_db=True)
        except Exception as e:
            logger.error(f"Failed to update markets and analyze: {e}", exc_info=True)
    
    
    def force_update(self) -> None:
        """强制立即更新（用于手动触发）"""
        logger.info("Force update triggered")
        self._update_markets_and_analyze()


# 全局单例
_polymarket_worker: Optional[PolymarketWorker] = None
_worker_lock = threading.Lock()


def get_polymarket_worker() -> PolymarketWorker:
    """获取PolymarketWorker单例"""
    global _polymarket_worker
    with _worker_lock:
        if _polymarket_worker is None:
            update_interval = int(os.getenv("POLYMARKET_UPDATE_INTERVAL_MIN", "30"))
            cache_minutes = int(os.getenv("POLYMARKET_ANALYSIS_CACHE_MIN", "30"))
            _polymarket_worker = PolymarketWorker(
                update_interval_minutes=update_interval,
                analysis_cache_minutes=cache_minutes
            )
        return _polymarket_worker



def main() -> None:
    """本地调试入口：仅执行一次 Polymarket 分析流程。"""
    parser = argparse.ArgumentParser(description="Run one-shot Polymarket analysis")
    parser.add_argument("--max-opportunities", type=int, default=30, help="Max opportunities to send to LLM")
    parser.add_argument("--per-category-limit", type=int, default=50, help="Fetch limit per category")
    parser.add_argument("--no-save", action="store_true", help="Do not save analysis result to DB")
    args = parser.parse_args()

    worker = PolymarketWorker()
    summary = worker.run_once_analysis(
        per_category_limit=args.per_category_limit,
        max_opportunities=args.max_opportunities,
        save_to_db=not args.no_save,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
