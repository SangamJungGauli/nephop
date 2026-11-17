pub mod random;
pub mod sampler;
pub mod space;
/// Disha (दिशा, "direction"): which way is "better"?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Disha {
    /// Fedi (फेदी, "foot of the hill"): lower y is better (for example, loss).
    Fedi,
    /// Chuchuro (चुचुरो, "peak"): higher y is better (for example, accuracy).
    Chuchuro,
}

/// English name for `Disha`.
pub type Direction = Disha;

/// English names, so `Direction::Minimize` and `Direction::Maximize` work.
#[allow(non_upper_case_globals)]
impl Disha {
    pub const Minimize: Disha = Disha::Fedi;
    pub const Maximize: Disha = Disha::Chuchuro;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_names_mean_the_same_disha() {
        assert_eq!(Direction::Minimize, Disha::Fedi);
        assert_eq!(Direction::Maximize, Disha::Chuchuro);
    }
}
