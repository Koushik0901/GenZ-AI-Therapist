'use client';

import { useEffect, useState } from 'react';

/**
 * Analytics Dashboard Component
 * Real-time system metrics, quality trends, and alerts
 */

interface DashboardMetrics {
  avg_response_quality: number;
  avg_response_time_ms: number;
  total_requests: number;
  error_rate: number;
  crisis_detections: number;
  regenerations: number;
  user_satisfaction_rate: number;
}

interface SystemAlert {
  id: string;
  alert_type: string;
  severity: 'info' | 'warning' | 'critical';
  message: string;
  created_at: string;
  acknowledged: boolean;
}

export default function AnalyticsDashboard() {
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [alerts, setAlerts] = useState<SystemAlert[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const [metricsRes, alertsRes] = await Promise.all([
          fetch('/api/metrics'),
          fetch('/api/alerts'),
        ]);

        if (metricsRes.ok) {
          const data = await metricsRes.json();
          setMetrics(data.metrics);
        }

        if (alertsRes.ok) {
          const data = await alertsRes.json();
          setAlerts(data.alerts);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load metrics');
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();

    // Refresh every 30 seconds
    const interval = setInterval(fetchMetrics, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="p-6 bg-white rounded-lg border border-neutral-200">
        <p className="text-neutral-600">Loading analytics...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 bg-red-50 rounded-lg border border-red-200">
        <p className="text-red-700">Error loading analytics: {error}</p>
      </div>
    );
  }

  if (!metrics) {
    return (
      <div className="p-6 bg-white rounded-lg border border-neutral-200">
        <p className="text-neutral-600">No metrics available yet</p>
      </div>
    );
  }

  const getHealthColor = (quality: number) => {
    if (quality >= 70) return 'text-green-600';
    if (quality >= 50) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getHealthBgColor = (quality: number) => {
    if (quality >= 70) return 'bg-green-50';
    if (quality >= 50) return 'bg-yellow-50';
    return 'bg-red-50';
  };

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-neutral-900">Analytics Dashboard</h1>
        <p className="text-sm text-neutral-600 mt-1">System performance and health metrics</p>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {/* Response Quality */}
        <div className={`p-4 rounded-lg border border-neutral-200 ${getHealthBgColor(metrics.avg_response_quality)}`}>
          <p className="text-xs text-neutral-600 mb-1">Avg Response Quality</p>
          <p className={`text-2xl font-bold ${getHealthColor(metrics.avg_response_quality)}`}>
            {Math.round(metrics.avg_response_quality)}
          </p>
          <p className="text-xs text-neutral-600 mt-2">/100</p>
        </div>

        {/* Response Time */}
        <div className="p-4 rounded-lg border border-neutral-200 bg-blue-50">
          <p className="text-xs text-neutral-600 mb-1">Avg Response Time</p>
          <p className="text-2xl font-bold text-blue-600">
            {Math.round(metrics.avg_response_time_ms)}
          </p>
          <p className="text-xs text-neutral-600 mt-2">ms</p>
        </div>

        {/* Total Requests */}
        <div className="p-4 rounded-lg border border-neutral-200 bg-purple-50">
          <p className="text-xs text-neutral-600 mb-1">Total Requests</p>
          <p className="text-2xl font-bold text-purple-600">{metrics.total_requests}</p>
          <p className="text-xs text-neutral-600 mt-2">processed</p>
        </div>

        {/* Error Rate */}
        <div className={`p-4 rounded-lg border border-neutral-200 ${metrics.error_rate > 5 ? 'bg-orange-50' : 'bg-green-50'}`}>
          <p className="text-xs text-neutral-600 mb-1">Error Rate</p>
          <p className={`text-2xl font-bold ${metrics.error_rate > 5 ? 'text-orange-600' : 'text-green-600'}`}>
            {metrics.error_rate.toFixed(1)}%
          </p>
          <p className="text-xs text-neutral-600 mt-2">of requests</p>
        </div>

        {/* Crisis Detections */}
        <div className="p-4 rounded-lg border border-neutral-200 bg-red-50">
          <p className="text-xs text-neutral-600 mb-1">Crisis Detections</p>
          <p className="text-2xl font-bold text-red-600">{metrics.crisis_detections}</p>
          <p className="text-xs text-neutral-600 mt-2">total</p>
        </div>

        {/* Regenerations */}
        <div className="p-4 rounded-lg border border-neutral-200 bg-amber-50">
          <p className="text-xs text-neutral-600 mb-1">Regenerations</p>
          <p className="text-2xl font-bold text-amber-600">{metrics.regenerations}</p>
          <p className="text-xs text-neutral-600 mt-2">quality retries</p>
        </div>

        {/* User Satisfaction */}
        <div className="p-4 rounded-lg border border-neutral-200 bg-green-50">
          <p className="text-xs text-neutral-600 mb-1">User Satisfaction</p>
          <p className="text-2xl font-bold text-green-600">
            {metrics.user_satisfaction_rate.toFixed(0)}%
          </p>
          <p className="text-xs text-neutral-600 mt-2">positive feedback</p>
        </div>
      </div>

      {/* Alerts Section */}
      <div className="mt-8">
        <h2 className="text-lg font-semibold text-neutral-900 mb-4">Active Alerts</h2>

        {alerts.length === 0 ? (
          <div className="p-4 rounded-lg border border-green-200 bg-green-50">
            <p className="text-sm text-green-700">✓ No active alerts - system is healthy!</p>
          </div>
        ) : (
          <div className="space-y-2">
            {alerts.map((alert) => (
              <div
                key={alert.id}
                className={`p-4 rounded-lg border-l-4 ${
                  alert.severity === 'critical'
                    ? 'bg-red-50 border-red-400'
                    : alert.severity === 'warning'
                      ? 'bg-yellow-50 border-yellow-400'
                      : 'bg-blue-50 border-blue-400'
                }`}
              >
                <div className="flex justify-between items-start">
                  <div>
                    <p className={`text-sm font-medium ${alert.severity === 'critical' ? 'text-red-700' : alert.severity === 'warning' ? 'text-yellow-700' : 'text-blue-700'}`}>
                      {alert.alert_type.replace(/_/g, ' ').toUpperCase()}
                    </p>
                    <p className="text-sm text-neutral-700 mt-1">{alert.message}</p>
                    <p className="text-xs text-neutral-600 mt-2">
                      {new Date(alert.created_at).toLocaleString()}
                    </p>
                  </div>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${alert.acknowledged ? 'bg-gray-100 text-gray-700' : 'bg-orange-100 text-orange-700'}`}>
                    {alert.acknowledged ? 'Acknowledged' : 'New'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="text-xs text-neutral-600 mt-8 pt-4 border-t border-neutral-200">
        <p>Last updated: {new Date().toLocaleTimeString()}</p>
        <p>Dashboard auto-refreshes every 30 seconds</p>
      </div>
    </div>
  );
}
