# GenAI-Canvas

# Dev Notes

> Basic steps to follow for all developers, while working on the code base.

There are three branches

- `main` - (This is the code that gets pushed to the prod)
 
   - `test` (This is the temp space, where all integrated features are placed to be tested.) Once complete and stable, it will be merged back to Main

        - `develop` (This is the temp space, where all integrated features are placed to be integrated.) Once complete and stable, it will be merged back to test
        
            - `{Feature Branch}` (Eg : "code_refactoring" Each developer working on a new use case, will have to do so in a new branch with a name reflecting the use case). Once complete and stable, it will be merged back to develop
