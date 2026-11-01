"""
统一表达式模板的编译和系数拟合模块
支持phase_idx参数的统一表达式模板
"""
import re
import numpy as np
from typing import Callable, Optional, Tuple, List
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from agent.imitation_based.llm_expression_compiler import ExpressionCompiler


class UnifiedExpressionCompiler(ExpressionCompiler):
    """统一表达式编译器，支持phase_idx参数"""
    
    def compile_unified_expression(self, expr_template: str, phase_idx: int) -> Tuple[Callable, List[str]]:
        """
        编译统一表达式模板，将phase_idx替换为具体值
        
        Args:
            expr_template: 统一表达式模板，包含phase_idx占位符
            phase_idx: 当前phase索引
        
        Returns:
            (compiled_func, coefficient_names) 元组
        """
        # 将phase_idx替换为具体值
        expr = expr_template.replace('phase_idx', str(phase_idx))
        
        # 使用父类的编译方法
        return self.compile_expression(expr)
    
    def fit_coefficients_for_phase(self,
                                   expr_template: str,
                                   phase_idx: int,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   initial_coeffs: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        为特定phase拟合系数
        
        Args:
            expr_template: 统一表达式模板
            phase_idx: phase索引
            X: 特征矩阵
            y: 目标值
            initial_coeffs: 初始系数
        
        Returns:
            (fitted_coeffs, mse) 元组
        """
        # 编译表达式（替换phase_idx）
        compiled_func, coeff_names = self.compile_unified_expression(expr_template, phase_idx)
        num_coeffs = len(coeff_names)
        
        # 定义损失函数
        def loss(coeffs):
            try:
                y_pred = compiled_func(X, *coeffs)
                if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                    return 1e10
                mse = mean_squared_error(y, y_pred)
                return mse
            except Exception:
                return 1e10
        
        # 初始化系数
        if initial_coeffs is None:
            initial_coeffs = np.random.randn(num_coeffs) * 0.1
        
        # 优化系数
        try:
            result = minimize(
                loss,
                initial_coeffs,
                method='BFGS',
                options={'maxiter': 1000}
            )
            
            if result.success:
                fitted_coeffs = result.x
                final_mse = loss(fitted_coeffs)
            else:
                # 如果BFGS失败，尝试L-BFGS-B
                result = minimize(
                    loss,
                    initial_coeffs,
                    method='L-BFGS-B',
                    bounds=[(-10, 10)] * num_coeffs,
                    options={'maxiter': 1000}
                )
                fitted_coeffs = result.x
                final_mse = loss(fitted_coeffs)
        except Exception as e:
            print(f"  Coefficient fitting failed for phase {phase_idx}: {e}, using random initialization")
            fitted_coeffs = initial_coeffs
            final_mse = loss(fitted_coeffs)
        
        return fitted_coeffs, final_mse
    
    def create_phase_predictor(self, expr_template: str, phase_idx: int, coeffs: np.ndarray) -> Callable:
        """
        为特定phase创建预测函数
        
        Args:
            expr_template: 统一表达式模板
            phase_idx: phase索引
            coeffs: 拟合的系数
        
        Returns:
            预测函数 f(X) -> y
        """
        compiled_func, _ = self.compile_unified_expression(expr_template, phase_idx)
        
        def predictor(X):
            return compiled_func(X, *coeffs)
        
        return predictor


class UnifiedLLMExpressionModel:
    """统一表达式模型，包含模板和所有phase的系数（支持pickle序列化）"""
    
    def __init__(self, expr_template: str, phase_coeffs: List[np.ndarray], compiler: UnifiedExpressionCompiler):
        """
        Args:
            expr_template: 统一表达式模板
            phase_coeffs: 每个phase的系数列表
            compiler: 统一表达式编译器
        """
        self.expr_template = expr_template
        self.phase_coeffs = phase_coeffs
        self.feature_names = compiler.feature_names  # 保存用于重建
        self.num_phase = len(phase_coeffs)
        self._predictors = None  # 延迟初始化
    
    def _get_predictors(self):
        """获取或创建预测器列表（延迟初始化）"""
        if self._predictors is None:
            compiler = UnifiedExpressionCompiler(self.feature_names)
            self._predictors = []
            for phase_idx in range(self.num_phase):
                predictor = compiler.create_phase_predictor(
                    self.expr_template, phase_idx, self.phase_coeffs[phase_idx]
                )
                self._predictors.append(predictor)
        return self._predictors
    
    def __getstate__(self):
        """序列化时排除不可pickle的对象"""
        state = self.__dict__.copy()
        state['_predictors'] = None
        return state
    
    def __setstate__(self, state):
        """反序列化时恢复状态"""
        self.__dict__.update(state)
        # predictors会在首次调用predict时延迟重建
    
    def predict(self, X: np.ndarray, phase_idx: Optional[int] = None) -> np.ndarray:
        """
        预测Q值
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            phase_idx: 如果指定，只预测该phase的Q值；否则返回所有phase的Q值
        
        Returns:
            如果phase_idx指定：返回该phase的Q值 (n_samples,)
            否则：返回所有phase的Q值 (n_samples, num_phase)
        """
        predictors = self._get_predictors()
        if phase_idx is not None:
            return predictors[phase_idx](X)
        else:
            # 返回所有phase的Q值
            predictions = [pred(X) for pred in predictors]
            return np.column_stack(predictions)
    
    def get_expression_for_phase(self, phase_idx: int) -> str:
        """获取特定phase的表达式（带系数）"""
        expr = self.expr_template.replace('phase_idx', str(phase_idx))
        
        # 提取系数名
        coeff_names = re.findall(r'\ba(\d+)\b', expr)
        coeff_names = sorted(set([f'a{num}' for num in coeff_names]), 
                           key=lambda x: int(x[1:]))
        
        # 替换系数
        for idx, coeff_name in enumerate(coeff_names):
            expr = re.sub(rf'\b{coeff_name}\b', f'{self.phase_coeffs[phase_idx][idx]:.6f}', expr)
        
        return expr
    
    def get_all_expressions(self) -> List[str]:
        """获取所有phase的表达式"""
        return [self.get_expression_for_phase(i) for i in range(self.num_phase)]

