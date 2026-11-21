//! Bounds supervisor waits for NVML while allowing at most one blocking call.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// A single in-flight NVML operation shared by startup, telemetry, and actuation.
#[derive(Clone, Default)]
pub(crate) struct NvmlGate(Arc<AtomicBool>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NvmlRunError {
    Busy,
    TimedOut,
    WorkerFailed,
}

impl NvmlRunError {
    pub(crate) const ALL: [Self; 3] = [Self::Busy, Self::TimedOut, Self::WorkerFailed];

    /// Stable metric label.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Busy => "gate_busy",
            Self::TimedOut => "deadline_exceeded",
            Self::WorkerFailed => "worker_failed",
        }
    }
}

pub(crate) struct Permit(Arc<AtomicBool>);

impl Drop for Permit {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

impl NvmlGate {
    pub(crate) fn try_enter(&self) -> Result<Permit, NvmlRunError> {
        self.0
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| Permit(Arc::clone(&self.0)))
            .map_err(|_| NvmlRunError::Busy)
    }

    #[cfg(test)]
    fn is_busy(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }

    /// Abandon the wait after `timeout`; the permit remains held until the
    /// blocking driver call actually returns. NVML offers no safe interruption.
    pub(crate) async fn run<T, F>(&self, timeout: Duration, operation: F) -> Result<T, NvmlRunError>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let permit = self.try_enter()?;
        let task = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            operation()
        });
        match tokio::time::timeout(timeout, task).await {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(_)) => Err(NvmlRunError::WorkerFailed),
            Err(_) => Err(NvmlRunError::TimedOut),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    #[tokio::test]
    async fn timed_out_read_does_not_spawn_another_blocked_worker() {
        let gate = NvmlGate::default();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let first = gate.run(Duration::from_millis(40), move || {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            1
        });
        let started = tokio::task::spawn_blocking(move || started_rx.recv().unwrap());
        assert_eq!(first.await, Err(NvmlRunError::TimedOut));
        started.await.unwrap();

        let (unexpected_tx, unexpected_rx) = mpsc::channel();
        assert_eq!(
            gate.run(Duration::from_millis(40), move || {
                unexpected_tx.send(()).unwrap();
                2
            })
            .await,
            Err(NvmlRunError::Busy)
        );
        assert!(unexpected_rx.try_recv().is_err());

        release_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while gate.is_busy() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(gate.run(Duration::from_millis(40), || 3).await, Ok(3));
    }

    #[tokio::test]
    async fn cancelling_waiter_keeps_gate_busy_until_driver_returns() {
        let gate = NvmlGate::default();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let pending = tokio::spawn({
            let gate = gate.clone();
            async move {
                gate.run(Duration::from_secs(30), move || {
                    started_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                })
                .await
            }
        });
        tokio::task::spawn_blocking(move || started_rx.recv().unwrap())
            .await
            .unwrap();
        pending.abort();
        assert_eq!(
            gate.run(Duration::from_millis(40), || 1).await,
            Err(NvmlRunError::Busy)
        );
        release_tx.send(()).unwrap();
    }
}
