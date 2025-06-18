
#[derive(PartialEq, Copy, Clone)]
pub enum RunMode {
    Pause,
    Stepping,
    Running
}

pub struct ExecutionObjects {
    pub iteration: usize,
    pub run_mode: RunMode,
}
pub enum Events {
    PauseRequested,
    SteppingRequested,
    PlayRequested,
}