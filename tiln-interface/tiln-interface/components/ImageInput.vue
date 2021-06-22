<template>
  <a-layout id="components-layout-demo-side" style="min-height: 100vh">
    <a-layout-sider v-model="collapsed" collapsible>
     <NuxtLink to="/">
        <div align="center" style="margin: 10px;">
            <img width="50px" src="~/assets/TOXIC.png" />
        </div>
      </NuxtLink>
      <a-menu theme="dark" :default-selected-keys="['1']" mode="inline">
        <a-sub-menu key="sub1">
          <span slot="title"><a-icon type="monitor" />
          <span>Test</span></span>
          <a-menu-item key="3">
            <NuxtLink to="/">text input</NuxtLink>
          </a-menu-item>
          <a-menu-item key="4">
            <NuxtLink to="/image">image</NuxtLink>
          </a-menu-item>
          <a-menu-item key="5">
            <NuxtLink to="/text-file">text file</NuxtLink>
          </a-menu-item>
        </a-sub-menu>
        <a-menu-item key="2">
          <a-icon type="info" />
          <NuxtLink to="/about"></NuxtLink>
        </a-menu-item>
      </a-menu>
    </a-layout-sider>
    <a-layout>
      <a-layout-header style="background: #fff; padding: 0" />
      <a-layout-content style="margin: 0 16px">
        <a-breadcrumb style="margin: 16px 0">
          <a-breadcrumb-item>Offensive Message Detection</a-breadcrumb-item>
        </a-breadcrumb>
        
        <a-select ref="modelselect" default-value="soft_voting_best3" style="width: 200px" @change="handleChange">
          <a-select-opt-group>
            <span slot="label"><a-icon type="usergroup-add" />Voting</span>
            <a-select-option value="soft_voting_best3">
              soft_voting_best3
            </a-select-option>
            <a-select-option value="hard_voting_best3">
              hard_voting_best3
            </a-select-option>
            <a-select-option value="soft_voting_all">
              soft_voting_all
            </a-select-option>
            <a-select-option value="hard_voting_all">
              hard_voting_all
            </a-select-option>
          </a-select-opt-group>
          <a-select-opt-group label="Singular">
            <a-select-option value="multi_naive_bayes_model">
              multi_naive_bayes_model
            </a-select-option>
            <a-select-option value="dtree_model">
              dtree_model
            </a-select-option>
            <a-select-option value="knn_model">
              knn_model
            </a-select-option>
            <a-select-option value="lr_model">
              lr_model
            </a-select-option>
            <a-select-option value="pa_model">
              pa_model
            </a-select-option>
            <a-select-option value="rforest_model">
              rforest_model
            </a-select-option>
            <a-select-option value="svm_model">
              svm_model
            </a-select-option>
          </a-select-opt-group>
        </a-select>

         <div>
           <div>
            <label>Imagine
                <input type="file" id="file" ref="file" v-on:change="handleFileUpload()"/>
            </label>
           </div>
                <a-button @click="trimite">Trimite</a-button>
         </div>
          
          <img :src="test_image_src" />
        
      </a-layout-content>
      <a-layout-footer style="text-align: center">
        Offensive Messsage Detection ©2021 Created by Toxic Players
      </a-layout-footer>
    </a-layout>
  </a-layout>

  

  
</template>
<script>
import axios from "axios"


export default {
  data() {
    return {
      data: [],

      comment: null,
      output: null,
      impact_words: null,
      collapsed: false,

      test_image_src: "#",
      
      myStyle:{
        backgroundColor:"#262626" 
      },

      myImg: "http://getdrawings.com/free-icon/tick-icon-52.png"
    }},
    
    methods: {
      handleChange(value) {
      console.log(`selected ${value}`);
    },
    handleFileUpload(){
        this.file = this.$refs.file.files[0];
    },
      trimite: function() {
      
      let formData = new FormData();
        formData.append('file', this.file);
        formData.append('model', this.$refs.modelselect.$el.innerText);
        axios.post( 'http://localhost:105/image',
        formData,
        {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        }
        ).then(response => {
          this.test_image_src = "data:image/png;base64," + response.data.file        
          console.log('SUCCESS!!');
          
        })
        .catch(function(){
        console.log('FAILURE!!');
        });
       
    },
    }
    }
</script>

<style scoped>
/* tile uploaded pictures */
.upload-list-inline >>> .ant-upload-list-item {
  float: left;
  width: 200px;
  margin-right: 8px;
}
.upload-list-inline >>> .ant-upload-animate-enter {
  animation-name: uploadAnimateInlineIn;
}
.upload-list-inline >>> .ant-upload-animate-leave {
  animation-name: uploadAnimateInlineOut;
}
</style>

<style>
#components-layout-demo-side .logo {
  height: 32px;
  background: rgba(255, 255, 255, 0.2);
  margin: 16px;
}
.upload-list-inline >>> .ant-upload-list-item {
  float: left;
  width: 200px;
  margin-right: 8px;
}
.upload-list-inline >>> .ant-upload-animate-enter {
  animation-name: uploadAnimateInlineIn;
}
.upload-list-inline >>> .ant-upload-animate-leave {
  animation-name: uploadAnimateInlineOut;
}
</style>

<style>
.container {
  margin: 0 auto;
  min-height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  text-align: center;
}

.title {
  font-family:
    'Quicksand',
    'Source Sans Pro',
    -apple-system,
    BlinkMacSystemFont,
    'Segoe UI',
    Roboto,
    'Helvetica Neue',
    Arial,
    sans-serif;
  display: block;
  font-weight: 300;
  font-size: 100px;
  color: #35495e;
  letter-spacing: 1px;
}

.subtitle {
  font-weight: 300;
  font-size: 42px;
  color: #526488;
  word-spacing: 5px;
  padding-bottom: 15px;
}

.links {
  padding-top: 15px;
}
</style>
