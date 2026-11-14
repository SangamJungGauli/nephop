pub mod random;
pub mod space;
/// Which way is "better"?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Lower y is better (for example, loss).
    Minimize,
    /// Higher y is better (for example, accuracy).
    Maximize,
}
