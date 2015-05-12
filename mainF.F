
      
      SubRoutine FortranTest(BlockDim, GridDim)
      Implicit None
      Integer BlockDim, GridDim
      Integer IOut
      Parameter (IOut = 6)
      
 1000 Format(' Enter Fotran function')
 1100 Format(' Leave Fotran function')
      
      Write(IOut, 1000)
      
      Call CalculatePiWithDevice(BlockDim, GridDim)
      
      Write(IOut, 1100)
      
      Call Flush(IOut)
      
      Return
      End
