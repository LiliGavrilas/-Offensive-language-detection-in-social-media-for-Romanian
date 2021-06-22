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
        </a-breadcrumb>
        
        <a-descriptions title="Offensive Language Detection">
            <a-descriptions-item label="Membri">
            Calmis Elena, Gavrilas Liliana, Plesescu Adrian
            </a-descriptions-item>
        </a-descriptions>

        <a-card hoverable style="width: 300px">
          <img
            slot="cover"
            alt="example"
            src="~/assets/TOXIC.png"
          />
          <template slot="actions" class="ant-card-actions">
            <a href="https://bit.ly/3o6aoa2" target="_blank"><a-icon key="github" type="github"/></a>
            <a href="https://bit.ly/3f854yZ" target="_blank"><a-icon key="project" type="project" /></a>
            <a href="shorturl.at/iCJV6" target="_blank"><a-icon key="file-done" type="file-done" /></a>
          </template>
          <a-card-meta title="Offensive Language Detection" description="Proiect TILN -> introduce un input text in romana pentru a primi un scor reprezentand cat de offensive este.">
            <a-avatar
              slot="avatar"
              src="https://www.uaic.ro/wp-content/uploads/2020/06/sigla-informatica-520x520.jpg"
            />
          </a-card-meta>
        </a-card>
        
      </a-layout-content>
      <a-layout-footer style="text-align: center">
        Offensive Messsage Detection ©2021 Created by Toxic Players
      </a-layout-footer>
    </a-layout>
  </a-layout>
</template>
<script>
export default {
  data() {
    return {
      defaultFileList: [
        {
          uid: '1',
          name: 'xxx.png',
          status: 'done',
          response: 'Server Error 500', // custom error message to show
          url: 'http://www.baidu.com/xxx.png',
        },
        {
          uid: '2',
          name: 'yyy.png',
          status: 'done',
          url: 'http://www.baidu.com/yyy.png',
        },
        {
          uid: '3',
          name: 'zzz.png',
          status: 'error',
          response: 'Server Error 500', // custom error message to show
          url: 'http://www.baidu.com/zzz.png',
        },
      ],
    };
  },
  methods: {
    handleChange({ file, fileList }) {
      if (file.status !== 'uploading') {
        console.log(file, fileList);
      }
    },
  },
};
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
