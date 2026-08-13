"""分类器测试 — ClassificationStep 与 ClassifierWrapper 的纯逻辑。

注意: 导入 src.ml.classifier_wrapper 需要 torch + transformers。
推理路径用假 wrapper 替代，避免在测试中加载真实模型。
"""

from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.database.models import Announcement, Base, Classification, Company
from src.ml.classifier_wrapper import ClassificationResult, ClassifierWrapper
from src.pipeline.classifier import ClassificationStep


class FakeWrapper:
    """替代 ClassifierWrapper 的假实现，返回固定结果。"""

    def __init__(self, results=None, **kwargs) -> None:
        self.results = results or [
            ClassificationResult(major_category="C", sub_category="buyback", confidence=0.95)
        ]
        self._sub_to_major = {}

    def classify_batch(self, texts: list[str]) -> list[ClassificationResult]:
        # 按输入长度循环返回固定结果
        return [self.results[i % len(self.results)] for i in range(len(texts))]

    def set_sub_to_major_mapping(self, mapping: dict[str, str]) -> None:
        self._sub_to_major = mapping


class TestClassifierWrapper:
    def test_get_major_category_from_mapping(self):
        """子类别→大类别的映射优先命中静态映射。"""
        w = object.__new__(ClassifierWrapper)  # 绕过 __init__，不加载真实模型
        w._sub_to_major = {"buyback": "C", "earnings_q1": "A"}
        assert w._get_major_category("buyback") == "C"
        assert w._get_major_category("earnings_q1") == "A"

    def test_get_major_category_fallback_to_registry(self):
        """映射缺失时回退到 event_registry（基于 event_types.yaml）。"""
        w = object.__new__(ClassifierWrapper)
        w._sub_to_major = {}
        assert w._get_major_category("litigation") == "F"  # 风险事件
        assert w._get_major_category("st_delisting_risk") == "F"

    def test_get_major_category_unknown(self):
        """完全未知的子类别返回 '?'。"""
        w = object.__new__(ClassifierWrapper)
        w._sub_to_major = {}
        assert w._get_major_category("no_such_label") == "?"


class TestClassificationStep:
    def _make_db(self):
        """创建独立的内存库（ClassificationStep 内部用 get_engine，需 patch 到它）。"""
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            company = Company(
                stock_code="000001.SZ", stock_name="平安银行", exchange="SZSE",
                industry="电气设备", is_tracked=True,
            )
            session.add(company)
            session.flush()
            ann = Announcement(
                company_id=company.id,
                announcement_id="ANN-CLS-001",
                title="中国宝安:关于回购公司股份方案的公告",
                full_text="公司拟以自有资金回购公司股份，回购金额不低于500万元。",
                published_date=date(2024, 4, 25),
                processing_status="preprocessed",
            )
            session.add(ann)
            session.commit()
        return engine

    def test_run_classifies_and_persists(self, tmp_path, monkeypatch):
        """run() 对 preprocessed 公告推理，落库 classifications 并更新状态。"""
        engine = self._make_db()

        monkeypatch.setattr("src.pipeline.classifier.get_engine", lambda: engine)
        fake = FakeWrapper([ClassificationResult("C", "buyback", 0.95)])
        monkeypatch.setattr("src.pipeline.classifier.ClassifierWrapper", lambda **kwargs: fake)

        step = ClassificationStep(model_path=str(tmp_path))
        count = step.run()

        assert count == 1
        with Session(engine) as session:
            cls = session.query(Classification).one()
            assert cls.major_category == "C"
            assert cls.sub_category == "buyback"
            assert cls.confidence == pytest.approx(0.95)
            # 行业从公司自带属性带入，不经模型
            assert cls.industry == "电气设备"
            assert cls.industry_group == "新能源与电力"

            ann = session.query(Announcement).one()
            assert ann.processing_status == "classified"

    def test_run_no_announcements_returns_zero(self, tmp_path, monkeypatch):
        """没有待分类公告时返回 0。"""
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)

        monkeypatch.setattr("src.pipeline.classifier.get_engine", lambda: engine)
        monkeypatch.setattr(
            "src.pipeline.classifier.ClassifierWrapper",
            lambda **kwargs: FakeWrapper(),
        )

        step = ClassificationStep(model_path=str(tmp_path))
        assert step.run() == 0

    def test_run_raises_without_trained_model(self):
        """无微调模型时应 fail-fast 抛错，而不是加载原始模型产垃圾分类。"""
        with pytest.raises(RuntimeError, match="分类模型未训练"):
            ClassificationStep(model_path="nonexistent/model/path")
