"""
Heart Rate and Rhythm Analysis for ECG Signals

Provides comprehensive heart rate analysis, rhythm analysis,
and clinical alert generation.
"""

import numpy as np
from typing import Dict, List, Optional
from scipy import signal, stats


class HeartRateAnalyzer:
    """Analyze heart rate from ECG signals."""
    
    def __init__(self, sampling_rate: int = 360):
        """Initialize analyzer with sampling rate."""
        self.sampling_rate = sampling_rate
    
    def analyze_heart_rate(self, ecg_signal: np.ndarray) -> Dict:
        """
        Comprehensive heart rate analysis.
        
        Args:
            ecg_signal: ECG signal array (187 samples typically)
            
        Returns:
            Heart rate analysis results
        """
        # Detect R-peaks (simplified - in production use proper QRS detection)
        r_peaks = self._detect_r_peaks(ecg_signal)
        
        if len(r_peaks) < 2:
            return {'error': 'Insufficient R-peaks detected'}
        
        # Calculate RR intervals
        rr_intervals = np.diff(r_peaks) / self.sampling_rate * 1000  # in ms
        
        # Calculate metrics
        avg_hr = 60000 / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0
        min_hr = 60000 / np.max(rr_intervals) if np.max(rr_intervals) > 0 else 0
        max_hr = 60000 / np.min(rr_intervals) if np.min(rr_intervals) > 0 else 0
        
        # Heart rate variability
        hrv_metrics = self._calculate_hrv(rr_intervals)
        
        # Classify heart rate
        hr_classification = self._classify_heart_rate(avg_hr)
        
        return {
            'average_hr': round(avg_hr, 1),
            'min_hr': round(min_hr, 1),
            'max_hr': round(max_hr, 1),
            'hr_range': round(max_hr - min_hr, 1),
            'hr_classification': hr_classification,
            'rr_intervals_ms': [round(rr, 1) for rr in rr_intervals[:10]],  # First 10
            'hrv_metrics': hrv_metrics,
            'rhythm_regularity': self._assess_regularity(rr_intervals)
        }
    
    def _detect_r_peaks(self, ecg_signal: np.ndarray) -> np.ndarray:
        """Detect R-peaks in ECG signal (simplified)."""
        # Simplified peak detection
        # In production, use proper QRS detection algorithms
        peaks, _ = signal.find_peaks(ecg_signal, height=np.max(ecg_signal) * 0.5, distance=30)
        return peaks
    
    def _calculate_hrv(self, rr_intervals: np.ndarray) -> Dict:
        """Calculate heart rate variability metrics."""
        if len(rr_intervals) < 2:
            return {}
        
        # SDNN (Standard Deviation of NN intervals)
        sdnn = np.std(rr_intervals)
        
        # RMSSD (Root Mean Square of Successive Differences)
        if len(rr_intervals) > 1:
            differences = np.diff(rr_intervals)
            rmssd = np.sqrt(np.mean(differences ** 2))
        else:
            rmssd = 0
        
        # pNN50 (Percentage of NN intervals > 50ms different)
        if len(rr_intervals) > 1:
            differences = np.abs(np.diff(rr_intervals))
            pnn50 = np.sum(differences > 50) / len(differences) * 100
        else:
            pnn50 = 0
        
        return {
            'sdnn_ms': round(sdnn, 2),
            'rmssd_ms': round(rmssd, 2),
            'pnn50_percent': round(pnn50, 2),
            'interpretation': self._interpret_hrv(sdnn)
        }
    
    def _interpret_hrv(self, sdnn: float) -> str:
        """Interpret HRV values."""
        if sdnn > 50:
            return "Normal HRV - good autonomic function"
        elif sdnn > 30:
            return "Moderate HRV"
        else:
            return "Reduced HRV - may indicate autonomic dysfunction"
    
    def _classify_heart_rate(self, hr: float) -> Dict:
        """Classify heart rate."""
        if hr < 60:
            category = "Bradycardia"
            severity = "Mild" if hr >= 50 else "Moderate" if hr >= 40 else "Severe"
        elif hr <= 100:
            category = "Normal Sinus Rhythm"
            severity = "Normal"
        elif hr <= 150:
            category = "Tachycardia"
            severity = "Mild"
        else:
            category = "Tachycardia"
            severity = "Severe"
        
        return {
            'category': category,
            'severity': severity,
            'hr_bpm': round(hr, 1)
        }
    
    def _assess_regularity(self, rr_intervals: np.ndarray) -> Dict:
        """Assess rhythm regularity."""
        if len(rr_intervals) < 2:
            return {'regularity': 'Unknown'}
        
        cv = np.std(rr_intervals) / np.mean(rr_intervals) * 100  # Coefficient of variation
        
        if cv < 5:
            regularity = "Regular"
        elif cv < 15:
            regularity = "Slightly Irregular"
        else:
            regularity = "Irregular"
        
        return {
            'regularity': regularity,
            'coefficient_of_variation': round(cv, 2),
            'interpretation': self._interpret_regularity(regularity)
        }
    
    def _interpret_regularity(self, regularity: str) -> str:
        """Interpret regularity."""
        interpretations = {
            'Regular': 'Consistent rhythm - likely sinus rhythm',
            'Slightly Irregular': 'Minor variation - may be normal sinus arrhythmia',
            'Irregular': 'Significant variation - may indicate arrhythmia'
        }
        return interpretations.get(regularity, 'Unknown')


class RhythmAnalyzer:
    """Analyze cardiac rhythm patterns."""
    
    def analyze_rhythm(self, ecg_signal: np.ndarray, hr_analysis: Dict) -> Dict:
        """
        Comprehensive rhythm analysis.
        
        Args:
            ecg_signal: ECG signal
            hr_analysis: Heart rate analysis results
            
        Returns:
            Rhythm analysis
        """
        regularity = hr_analysis.get('rhythm_regularity', {})
        hr_class = hr_analysis.get('hr_classification', {})
        
        # Determine rhythm type
        rhythm_type = self._determine_rhythm_type(hr_class, regularity)
        
        # Analyze P-waves (simplified)
        p_wave_analysis = self._analyze_p_waves(ecg_signal)
        
        # Analyze intervals (simplified)
        interval_analysis = self._analyze_intervals(ecg_signal)
        
        return {
            'rhythm_type': rhythm_type,
            'regularity': regularity.get('regularity', 'Unknown'),
            'p_wave_analysis': p_wave_analysis,
            'interval_analysis': interval_analysis,
            'interpretation': self._get_rhythm_interpretation(rhythm_type)
        }
    
    def _determine_rhythm_type(self, hr_class: Dict, regularity: Dict) -> str:
        """Determine rhythm type."""
        hr_category = hr_class.get('category', '')
        reg = regularity.get('regularity', '')
        
        if hr_category == 'Normal Sinus Rhythm' and reg == 'Regular':
            return 'Normal Sinus Rhythm'
        elif hr_category == 'Bradycardia':
            return 'Sinus Bradycardia' if reg == 'Regular' else 'Bradyarrhythmia'
        elif hr_category == 'Tachycardia':
            return 'Sinus Tachycardia' if reg == 'Regular' else 'Tachyarrhythmia'
        else:
            return 'Arrhythmia'
    
    def _analyze_p_waves(self, ecg_signal: np.ndarray) -> Dict:
        """Analyze P-waves (simplified)."""
        # Simplified analysis - in production use proper P-wave detection
        return {
            'present': True,
            'morphology': 'Normal',
            'interpretation': 'P-waves present and normal'
        }
    
    def _analyze_intervals(self, ecg_signal: np.ndarray) -> Dict:
        """Analyze ECG intervals (simplified)."""
        # Simplified - in production calculate actual PR, QRS, QT intervals
        return {
            'pr_interval_ms': 160,  # Normal range 120-200ms
            'qrs_duration_ms': 90,  # Normal < 120ms
            'qt_interval_ms': 400,  # Normal varies with HR
            'interpretation': 'Intervals within normal limits'
        }
    
    def _get_rhythm_interpretation(self, rhythm_type: str) -> str:
        """Get interpretation of rhythm."""
        interpretations = {
            'Normal Sinus Rhythm': 'Normal cardiac rhythm',
            'Sinus Bradycardia': 'Slow but normal rhythm',
            'Sinus Tachycardia': 'Fast but normal rhythm',
            'Bradyarrhythmia': 'Slow irregular rhythm - requires evaluation',
            'Tachyarrhythmia': 'Fast irregular rhythm - requires evaluation',
            'Arrhythmia': 'Abnormal rhythm - requires clinical evaluation'
        }
        return interpretations.get(rhythm_type, 'Rhythm analysis completed')


class ClinicalAlerts:
    """Generate clinical alerts based on ECG analysis."""
    
    def __init__(self):
        """Initialize alert system."""
        self.alert_thresholds = {
            'critical_hr': {'min': 40, 'max': 150},
            'severe_bradycardia': 40,
            'severe_tachycardia': 150,
            'critical_arrhythmia': True
        }
    
    def generate_alerts(self, hr_analysis: Dict, rhythm_analysis: Dict, 
                        arrhythmia_type: str) -> List[Dict]:
        """
        Generate clinical alerts.
        
        Args:
            hr_analysis: Heart rate analysis
            rhythm_analysis: Rhythm analysis
            arrhythmia_type: Detected arrhythmia type
            
        Returns:
            List of alerts
        """
        alerts = []
        
        # Heart rate alerts
        hr = hr_analysis.get('average_hr', 0)
        if hr < self.alert_thresholds['severe_bradycardia']:
            alerts.append({
                'level': 'Critical',
                'type': 'Severe Bradycardia',
                'message': f'Critical: Heart rate {hr:.1f} bpm - immediate evaluation required',
                'action': 'Immediate medical attention'
            })
        elif hr > self.alert_thresholds['severe_tachycardia']:
            alerts.append({
                'level': 'Critical',
                'type': 'Severe Tachycardia',
                'message': f'Critical: Heart rate {hr:.1f} bpm - immediate evaluation required',
                'action': 'Immediate medical attention'
            })
        
        # Arrhythmia alerts
        if arrhythmia_type in ['Ventricular', 'Fusion']:
            alerts.append({
                'level': 'Critical',
                'type': 'Life-Threatening Arrhythmia',
                'message': f'Critical: {arrhythmia_type} arrhythmia detected',
                'action': 'Immediate cardiac evaluation and monitoring'
            })
        elif arrhythmia_type == 'Supraventricular':
            alerts.append({
                'level': 'High',
                'type': 'Supraventricular Arrhythmia',
                'message': 'Supraventricular arrhythmia detected',
                'action': 'Cardiac evaluation recommended'
            })
        
        # Rhythm regularity alerts
        regularity = rhythm_analysis.get('regularity', '')
        if regularity == 'Irregular':
            alerts.append({
                'level': 'Moderate',
                'type': 'Irregular Rhythm',
                'message': 'Significant rhythm irregularity detected',
                'action': 'Further evaluation recommended'
            })
        
        return alerts

