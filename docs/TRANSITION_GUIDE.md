---

## 14. Summary: What to Keep, What to Change

### 14.1 Keep (Copy Directly from BFSNet)

✅ **utils/gating.py** - `GrowthPolicyNet` class (identical!)  
✅ **REINFORCE algorithm** - Policy gradient logic (identical!)  
✅ **Pooling functions** - `_pool_nodes()` (identical!)  
✅ **Checkpoint format** - Save/load logic (minimal changes)  
✅ **Docker infrastructure** - Build/run commands (adapt volumes)  
✅ **Testing philosophy** - Unit + integration (same approach)  

### 14.2 Adapt (Modify from BFSNet)

🔄 **BFSNet → BFSLanguageCell** - Add recurrent processing  
🔄 **Single pass → Sequential** - Loop over tokens  
🔄 **Image loader → Text loader** - Different data format  
🔄 **Accuracy → Perplexity** - Different metric  
🔄 **Configs** - Add vocab_size, seq_len, etc.  

### 14.3 Add (New for BoeNet)

🆕 **Tokenization** - CharTokenizer, BPETokenizer  
🆕 **Text generation** - Autoregressive sampling  
🆕 **Perplexity tracking** - Language model metric  
🆕 **Sequence processing** - Hidden state management  
🆕 **Gradient clipping** - Essential for RNNs  

---

## 15. Next Steps

After completing this transition:

1. **Validate Architecture**: Test BFSLanguageCell on dummy data
2. **Train on Shakespeare**: Full Phase 1 training (10 epochs)
3. **Tune Threshold**: Use `--debug_policy` to measure grow_prob
4. **Compare to LSTM**: Implement baseline, compare perplexity
5. **Generate Text**: Qualitative evaluation of generation quality
6. **Document Results**: Write Phase 1 report
7. **Plan Phase 2**: Prepare for word-level (TinyStories)

---

**Document Version**: 1.0  
**Last Updated**: December 20, 2025  
**Status**: Complete - Ready for Implementation  
**Next**: Begin BoeNet implementation following this guide

**⚠️ Proprietary Software**: This project is closed source. All rights reserved.