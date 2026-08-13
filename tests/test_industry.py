"""行业划分注册表测试 — IndustryRegistry（config/industries.yaml）。"""

import pytest

from src.config import industry_registry


class TestIndustryRegistry:
    def test_resolve_hit(self):
        """命中映射返回对应行业域。"""
        assert industry_registry.resolve("电气设备") == "新能源与电力"
        assert industry_registry.resolve("半导体") == "科技"
        assert industry_registry.resolve("化学制药") == "医药生物"
        assert industry_registry.resolve("银行") == "金融"

    def test_resolve_unmapped_falls_back_to_default(self):
        """未命中映射归 default_group。"""
        assert industry_registry.resolve("全新行业") == "其他"

    def test_resolve_none_falls_back_to_default(self):
        """空行业归 default_group。"""
        assert industry_registry.resolve(None) == "其他"
        assert industry_registry.resolve("") == "其他"

    def test_default_group_matches_config(self):
        """default_group 应非空且为合法字符串。"""
        assert industry_registry.default_group

    def test_full_coverage_of_config_mapping(self):
        """配置内每个行业都能解析到非默认组（保证无错别字悬空）。"""
        for group, industries in industry_registry.groups.items():
            for industry in industries:
                assert industry_registry.resolve(industry) == group

    def test_no_duplicate_industry_across_groups(self):
        """同一行业不应映射到多个组。"""
        seen = {}
        for group, industries in industry_registry.groups.items():
            for industry in industries:
                assert industry not in seen, f"{industry} 出现在 {seen[industry]} 和 {group}"
                seen[industry] = group
